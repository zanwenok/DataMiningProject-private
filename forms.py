from flask.ext.wtf import Form
from wtforms import StringField, BooleanField
from wtforms.validators import DataRequired

class LoginForm(Form):
    openid = StringField('openid', validators=[DataRequired()])
    remember_me = BooleanField('remember_me', default=False)

class KnnForm(Form):
    k_knn = StringField('openid', validators=[DataRequired()])
    lp_knn = StringField('openid', validators=[DataRequired()])
    i_knn = StringField('openid', validators=[DataRequired()])