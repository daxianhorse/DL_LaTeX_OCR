import os

from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed
from wtforms import StringField, PasswordField, BooleanField, SubmitField, FloatField, TextAreaField, FileField
from wtforms.validators import DataRequired
# from flask_login import *
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

from predict import get_latex

from flask_bootstrap import Bootstrap

# UPLOAD_FOLDER = os.path.join('static', 'gravatar')

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.debug = True
app.secret_key = os.getenv('SECRET_KEY', 'secret string')
bootstrap = Bootstrap(app=app)


class uploadForm(FlaskForm):
    file_upload = FileField('上传图片')
    submit = SubmitField('提交')


@app.route('/file/<path:path>')
def send_js(path):
    return send_from_directory('.cache', path)


@app.route('/', methods=['GET', 'POST'])
def transform():
    # return render_template('login.html')
    form = uploadForm()
    if form.validate_on_submit():
        filename = form.file_upload.data.filename
        if not os.path.exists(os.path.join(app.root_path, ".cache")):
            os.mkdir(os.path.join(app.root_path, ".cache"))
        path = os.path.join(app.root_path, ".cache", filename)
        form.file_upload.data.save(path)
        latex = get_latex(path)
        return render_template('transform.html', form=form, path=filename, latex=latex)
    return render_template('transform.html', form=form, path="0_2.png", latex="")


if __name__ == '__main__':
    app.run(debug=True)
