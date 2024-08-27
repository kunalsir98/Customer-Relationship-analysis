from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            Year_Birth=float(request.form.get('Year_Birth')),
            Income=float(request.form.get('Income')),
            Kidhome=int(request.form.get('Kidhome')),
            Teenhome=int(request.form.get('Teenhome')),
            Recency=int(request.form.get('Recency')),
            MntWines=float(request.form.get('MntWines')),
            MntFruits=float(request.form.get('MntFruits')),
            MntMeatProducts=float(request.form.get('MntMeatProducts')),
            MntFishProducts=float(request.form.get('MntFishProducts')),
            MntSweetProducts=float(request.form.get('MntSweetProducts')),
            MntGoldProds=float(request.form.get('MntGoldProds')),
            NumDealsPurchases=int(request.form.get('NumDealsPurchases')),
            NumWebPurchases=int(request.form.get('NumWebPurchases')),
            NumCatalogPurchases=int(request.form.get('NumCatalogPurchases')),
            NumStorePurchases=int(request.form.get('NumStorePurchases')),
            NumWebVisitsMonth=int(request.form.get('NumWebVisitsMonth')),
            AcceptedCmp3=int(request.form.get('AcceptedCmp3')),
            AcceptedCmp4=int(request.form.get('AcceptedCmp4')),
            AcceptedCmp5=int(request.form.get('AcceptedCmp5')),
            AcceptedCmp1=int(request.form.get('AcceptedCmp1')),
            AcceptedCmp2=int(request.form.get('AcceptedCmp2')),
            Complain=int(request.form.get('Complain')),
            Z_CostContact=int(request.form.get('Z_CostContact')),
            Z_Revenue=int(request.form.get('Z_Revenue')),
            Education=request.form.get('Education'),
            Marital_Status=request.form.get('Marital_Status')
        )
        
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)
        
        # Assuming the model's prediction output is categorical (e.g., '1' for positive response and '0' for negative)
        result = 'Positive Response' if pred[0] == 1 else 'Negative Response'
        
        return render_template('results.html', final_result=result)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
