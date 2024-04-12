from flask import Flask, flash, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import pandas as pd
from models.src import run_model
from models.src import visualize
from werkzeug.utils import secure_filename
import librosa
import glob 
# check

UPLOAD_FOLDER = './static/audio'
NON_STATIC_PATH = UPLOAD_FOLDER.split("static/")[1]
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///testv2.db'
app.config['SECRET_KEY'] = os.urandom(12)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)
global file_location

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable = False)
    true_label = db.Column(db.String(200), nullable=True)
    top1 = db.Column(db.String(200), nullable=False)
    model_used = db.Column(db.String(200), nullable=False)
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

@app.route('/job_setup/', methods=['POST'])
def job_setup():
    if request.method == 'POST':
        query = request.files['audio']
        if query.filename == '':
            flash('No file selected')
            return redirect('/')
        if not allowed_file(query.filename):
            flash('Need wav file')
            return redirect('/')
        
        # save file
        filename = secure_filename(query.filename)
        loc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        query.save(loc)
        
        global file_location
        file_location = loc
        
        # present choices
        model_paths = [file for file in glob.glob(f"models/output/**/*.joblib", recursive = True)]
        model_paths = [i.split("/")[-2] for i in model_paths]
        model_names = [i.split("_")[0] for i in model_paths]
        
        # we have model_names models ready
        # retreive performance score on test set
        test_results = pd.read_csv("models/output/test_results.csv", index_col = 0)
        test_results = test_results.reset_index(drop = True)
        tr = test_results.to_html(justify = "center")
        for i in model_names:
            tr = tr.replace(i, f"<a href='/job/{i}'>{i}</a>")
        return render_template('job-setup.html', test_results = tr, path = NON_STATIC_PATH+"/"+filename)
    else:
        return 'GET request to job_setup'
    
@app.route('/job/', methods=['GET'])
@app.route('/job/<model_type>', methods=['GET'])
def job(model_type = None):
    if model_type is None:
        
        id = int(request.args.get("id"))
        run = Result.query.get_or_404(id)
        features = [file for file in glob.glob(f"static/image/*.png", recursive = True)]
        features = [i.split("static/")[1] for i in features]
        return render_template('job.html', detail = run,
                                path = NON_STATIC_PATH+"/"+run.name, features = features)
    
    else:
        global file_location
        audio, sr = librosa.load(file_location)
        filename = file_location.split("/")[-1]

        save_location = os.path.join(app.config['UPLOAD_FOLDER'], f"../image")

        # features
        features = []
        features.append(visualize.harmonics_and_perceptrual(audio, sr, save_location,filename))
        features.append(visualize.mel_spectogram(audio, sr, save_location,filename))
        features.append(visualize.chroma_stft(audio, sr, save_location,filename))

        features = [i.split("static/")[1] for i in features]

        # prediction
        prediction = run_model.run(file_location, location = f"models/output/{model_type}_30sec",
                                  model_location = f"models/output/{model_type}_30sec/classifier.joblib",
                                  data_cache_location = f"models/output/{model_type}_30sec/data_cache.pkl")


        ## TODO: use threading to make this run while intermediate page displays
        run_detail = Result(name = filename,
                            top1 = prediction[0],
                            model_used = model_type)

        try:
            db.session.add(run_detail)
            db.session.commit()
            return render_template('job.html', detail = run_detail,
                                    path = NON_STATIC_PATH+"/"+filename, features = features)
        except Exception as e:
            return f'{e} There was an error displaying the job'


if __name__ == "__main__":
    app.run(debug=True)
