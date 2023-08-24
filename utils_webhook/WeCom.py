import hashlib
import requests
import base64


def calculate_md5_hash_of_file(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as file:
        while chunk := file.read(8192):  # 每次读取8192字节
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def base64_pic(file_path):
    with open(file_path, "rb") as image_file:
        image_data = image_file.read()

    # 进行Base64编码
    base64_encoded = base64.b64encode(image_data).decode('utf-8')
    return base64_encoded


class WeCom:
    def __init__(self, url: str):
        self.url = url
        self.content = None

    def generate_text(self, content: str):
        self.content = {
            "msgtype": "text",
            "text": {
                "content": content,
            }
        }

    def generate_md(self, content: str):
        self.content = {
            "msgtype": "markdown",
            "markdown": {
                "content": content
            }
        }

    def generate_img(self, file_path):
        self.content = {
            "msgtype": "image",
            "image": {
                "base64": base64_pic(file_path),
                "md5": calculate_md5_hash_of_file(file_path)
            }
        }

    def send(self):
        headers = {
            "Content-Type": "application/json"
        }  # http数据头，类型为json
        r = requests.post(self.url, headers=headers, json=self.content)  # 利用requests库发送post请求
        print(r)
        print("Send successfully!")


if __name__ == '__main__':
    weCom = WeCom(url="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=a7f64d24-662b-46ba-bbaf-cdf3f6209727")

    weCom.generate_img("G:\\git_G\\NewUserPredict\\photo\\unknown_2023_08_23_16_09.png")
    weCom.send()
    print("send ok")
