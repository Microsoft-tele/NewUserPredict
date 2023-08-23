# python 3.8
import time
import hmac
import hashlib
import base64
import urllib.parse
import requests

timestamp = str(round(time.time() * 1000))
secret = 'SEC04708ed4a3d0e6aa658bbfd6c186f41e7f20e14a8271edabb3ec7e78d1c0854b'
secret_enc = secret.encode('utf-8')
string_to_sign = '{}\n{}'.format(timestamp, secret)
string_to_sign_enc = string_to_sign.encode('utf-8')
hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))

print(timestamp)
print(sign)


def send2ding(content):
    url = f"https://oapi.dingtalk.com/robot/send?" \
          f"access_token=2a2b747af09906268b6cdbcc32462ecfe42ecebd0251591012f481b88cd2bed8&" \
          f"timestamp={timestamp}&" \
          f"sign={sign}"  # 这里就是群机器人的Webhook地址

    headers = {"Content-Type": "application/json"}  # http数据头，类型为json
    data = {
        "msgtype": "text",
        "text": {
            "content": f"{content}",
        },
        "at":{
            "isAll": True,
        }
    }
    r = requests.post(url, headers=headers, json=data)  # 利用requests库发送post请求


# 调用一下
if __name__ == "__main__":

    send2ding("From winwods")
# # 发送带有样式的消息，也可以加入链接进行跳转
#  send_weixin("实时新增用户反馈<font color=\"warning\">132例</font>，请相关同事注意。\n
#          >类型:<font color=\"comment\">用户反馈</font>
#          >普通用户反馈:<font color=\"comment\">117例</font>
#          >VIP用户反馈:<font color=\"comment\">15例</font>")
