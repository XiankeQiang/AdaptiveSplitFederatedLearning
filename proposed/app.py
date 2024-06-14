from flask import Flask, send_file, jsonify
import base64
import pandas as pd

app = Flask(__name__)

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()
        # 将图片数据编码为 base64 字符串
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    return image_base64

def get_table_base64(table_path):
    # 读取表格数据
    table_data = pd.read_csv(table_path)
    # 将表格数据转换为 HTML 表格字符串
    table_html = table_data.to_html(index=False)
    # 将 HTML 表格字符串编码为 base64 字符串
    table_base64 = base64.b64encode(table_html.encode('utf-8')).decode('utf-8')
    return table_base64

@app.route('/get_train_acc')
def get_train_acc():
    return get_image_base64("data/train_acc.png")

@app.route('/get_test_acc')
def get_test_acc():
    return get_image_base64("data/test_acc.png")

@app.route('/get_table_data')
def get_table_data():
    return get_table_base64("data/table.csv")

@app.route('/')
def index():
    return send_file('templates/index.html')

if __name__ == '__main__':
    app.run(port=8080)
