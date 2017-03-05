from __future__ import print_function
from flask import Flask,render_template,request,url_for,redirect,send_from_directory

import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import reader
import model
import time
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'do you guess,haha..dsdsddi4985'


UPLOAD_FOLDER='upfile/'
ALLOWED_EXTENSIONS=set(['png','jpg','jpeg'])

app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_string("model_file", "models.ckpt", "")
tf.app.flags.DEFINE_string("image_file", "a.jpg", "")
FLAGS = tf.app.flags.FLAGS

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    #return '<h1>it work</h1>'
    return render_template('index.html')

@app.route('/transform_photo_style', methods=['GET', 'POST'])
def deal_photo():
     modelDict = {'cubist':'cubist.ckpt-done',
                  'denoised_starry':'denoised_starry.ckpt-done',
                  'feathers':'feathers.ckpt-done',
                  'mosaic':'mosaic.ckpt-done',
                  'scream':'scream.ckpt-done',
                  'udnie':'udnie.ckpt-done',
                  'wave':'wave.ckpt-done',
                  }
     if request.method=='POST':
        file=request.files['pic']
        theme = request.form['theme']
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
            model_file = 'wave.ckpt-done'
            if theme!='':
                if modelDict[theme]!='':
                    model_file = modelDict[theme]
            style_transform('models/'+model_file,os.path.join(app.config['UPLOAD_FOLDER'])+file.filename,file.filename)
            return redirect('/uploads/'+file.filename)
        return 'transform error:file format error'
     return 'transform error:method not post'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('generated/',filename)

def style_transform(model_file,img_file,result_file):
    height = 0
    width = 0
    with open(img_file, 'rb') as img:
        with tf.Session().as_default() as sess:
            if img_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:
            image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)
            image = reader.get_image(img_file, height, width, image_preprocessing_fn)
            image = tf.expand_dims(image, 0)
            generated = model.net(image, training=False)
            generated = tf.squeeze(generated, [0])
            saver = tf.train.Saver(tf.global_variables())
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            FLAGS.model_file = os.path.abspath(model_file)
            saver.restore(sess, FLAGS.model_file)

            start_time = time.time()
            generated = sess.run(generated)
            generated = tf.cast(generated, tf.uint8)
            end_time = time.time()
            tf.logging.info('Elapsed time: %fs' % (end_time - start_time))
            generated_file = 'generated/'+result_file
            if os.path.exists('generated') is False:
                os.makedirs('generated')
            with open(generated_file, 'wb') as img:
                img.write(sess.run(tf.image.encode_jpeg(generated)))
                tf.logging.info('Done. Please check %s.' % generated_file)

if __name__ == '__main__':
    app.run(debug=True)