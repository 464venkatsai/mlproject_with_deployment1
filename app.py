from flask import Flask,render_template,request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline
app = Flask(__name__) 

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/prediction',methods=['GET','POST'])
def predict_data_points():
    if request.method == 'GET':
        return render_template('home.html')
    else : 
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch
            =request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            math_score=request.form.get('math_score'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score')
            
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        return render_template('home.html',result=result[0])
    
if __name__ == '__main__':
    app.run(debug=True)
