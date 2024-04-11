from flask import Flask, flash, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
from models.src import run_model
from werkzeug.utils import secure_filename
# check

UPLOAD_FOLDER = './static/audio'
NON_STATIC_PATH = UPLOAD_FOLDER.split("static/")[1]
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///testv1.db'
app.config['SECRET_KEY'] = os.urandom(12)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable = False)
    true_label = db.Column(db.String(200), nullable=True)
    top1 = db.Column(db.String(200), nullable=False)
    top3 = db.Column(db.String(200), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"" #{id} {true_label} {top1} {top3} {date_created}"
    
with app.app_context():
    db.create_all()

@app.route('/', methods=['GET'])
def index():
    history = Result.query.order_by(Result.date_created).all()
    return render_template('index.html', history=history)


@app.route('/delete/<int:id>')
def delete(id):
    run_to_delete = Result.query.get_or_404(id)

    try:
        db.session.delete(run_to_delete)
        db.session.commit()
        return redirect('/')
    except:
        return 'There was a problem deleting that run'

@app.route('/job/label/<int:id>', methods=['POST'])
def addlabel(id):
    run = Result.query.get_or_404(id)
    run.true_label = request.form['label']
    id = run.id
    try:
        db.session.commit()
        return redirect(f'/job/?id={id}')
    except:
        return 'There was a problem adding the label'


@app.route('/job/', methods=['POST', 'GET'])
def job():
    if request.method == 'POST':
        query = request.files['audio']
        if query.filename == '':
            flash('No file selected')
            return redirect('/')
        if not allowed_file(query.filename):
            flash('Need wav file')
            return redirect('/')

        filename = secure_filename(query.filename)
        query.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        prediction = run_model.run(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        ## TODO: use threading to make this run while intermediate page displays
        run_detail = Result(name = filename,
                            top1 = prediction[0],
                            top3 = ",".join(prediction))

        try:
            db.session.add(run_detail)
            db.session.commit()
            return render_template('job.html', detail = run_detail,
                                    path = NON_STATIC_PATH+"/"+filename)
        except Exception as e:
            return f'{e} There was an error displaying the job'
    else:
        id = int(request.args.get("id"))
        run = Result.query.get_or_404(id)
        return render_template('job.html', detail = run,
                                path = NON_STATIC_PATH+"/"+run.name)


if __name__ == "__main__":
    app.run(debug=True)
