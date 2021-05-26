from flask import Flask, request, render_template
from pyModules import firstOP
from pyModules import graph
import os
app = Flask(__name__)

@app.route('/')
def firstPage():
    filename = os.listdir('static/graph/')[0]
    grap = 'graph/'+filename
    return render_template('index.html', grap=grap)

@app.route('/', methods=['POST'])
def Input():
    query=""
    if request.form['submitButton']=='Search':
        query = request.form['query']
        grap=graph.Query_graph(query)
        category = request.form['Category']
        dictn, noType = firstOP.Query(query,category)
        if dictn==0:
            return render_template('nodata.html')
        else:
            return render_template('index1.html', data=dictn, grap=grap)
    elif request.form['submitButton']!='Search':
        lst=[]
        lst.append(request.form['submitButton'])
        print(lst)
        grap=graph.Query_graph(lst[0])
        return render_template('index.html', grap=grap)
        #return render_template('index1.html',lst[0])

if __name__ == "__main__":
    app.run(debug=True)
