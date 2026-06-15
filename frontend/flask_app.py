from flask import Flask, render_template
import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

@app.route('/')
def index():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
