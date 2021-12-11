import os
from flask import Flask, render_template, send_from_directory, request
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed
from wtforms import BooleanField, SubmitField, FileField

from predict import get_latex
from predict_biology import get_latex_biology

from flask_bootstrap import Bootstrap

# UPLOAD_FOLDER = os.path.join('static', 'gravatar')

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.debug = True
app.secret_key = os.getenv('SECRET_KEY', 'secret string')
bootstrap = Bootstrap(app=app)


class uploadForm(FlaskForm):
    file_upload = FileField('选择图片')
    pattern = BooleanField('切换为生物识别')
    submit = SubmitField('上传并开始识别')


@app.route('/file/<path:path>')
def send_js(path):
    return send_from_directory('.cache', path)


@app.route('/', methods=['GET', 'POST'])
def transform():
    # return render_template('login.html')
    print('shit')
    form = uploadForm()
    if request.method == 'POST':
        print('-----------')
        filename = form.file_upload.data.filename
        if not os.path.exists(os.path.join(app.root_path, ".cache")):
            os.mkdir(os.path.join(app.root_path, ".cache"))
        path = os.path.join(app.root_path, ".cache", filename)
        form.file_upload.data.save(path)
        latex = ""
        print('latex is here')
        if form.pattern:
            print('yes')
            latex += get_latex_biology(path)
        else:
            print('no')
            latex += get_latex(path)
        return render_template('transform.html', form=form, path=filename, latex=latex)
    return render_template('transform.html', form=form, path="0_2.png", latex="未开始识别")


if __name__ == '__main__':
    app.run(debug=True)
