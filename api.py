from flask import Flask, json, request, jsonify
import os
import urllib.request
from werkzeug.utils import secure_filename
import SemanticSearchPDF
 
app = Flask(__name__)
 

 
@app.route('/')
def main():
    return 'Homepage'
 
# Fetching the query performing operations using it and returning it back to the UI
@app.route('/fetch_query', methods=['POST'])
def fetch_query():
 
    data = request.get_json()
    text = data.get('text')
    response = SemanticSearchPDF.search_docs(text,"C:/Users/suashis.chakraborty/Downloads/FASB_equity.pdf")
    return jsonify(response)

#  Saving the pdf in local
@app.route('/upload', methods=['POST'])
def upload_file():
    
    if 'pdfFile' not in request.files:
        return {'message': 'No file selected'},400
    pdfFile = request.files['pdfFile']
    if pdfFile.filename == '':
        return {'message': 'No file selected'},400
    pdfFile.save('pdfs/'+pdfFile.filename)
    return {'message':'File uploaded successfully'}

if __name__ == '__main__':
    app.run(debug=True)