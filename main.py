from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    ImageMessage, ImageSendMessage
)
import pya3rt
import torch
import torchvision.transforms as transforms

from PIL import Image
from efficientnet_pytorch import EfficientNet
import os, shutil, re
import json
import wikipedia
import jaconv, cotohappy, romkan, pykakasi

app = Flask(__name__)

#-------------------------------------------------------------------------------
YOUR_CHANNEL_ACCESS_TOKEN = os.environ["YOUR_CHANNEL_ACCESS_TOKEN"]
YOUR_CHANNEL_SECRET = os.environ["YOUR_CHANNEL_SECRET"]
ACCESS_TOKEN_PUBLISH_URL = 'https://api.ce-cotoha.com/v1/oauth/accesstokens'
API_BASE_URL = 'https://api.ce-cotoha.com/api/dev/'
CLIENT_ID = os.environ["CLIENT_ID"]
CLIENT_SECRET = os.environ["CLIENT_SECRET"]
PYART_API_KEY = os.environ["PYART_API_KEY"]
coy = cotohappy.API(ACCESS_TOKEN_PUBLISH_URL,
                    API_BASE_URL,
                    CLIENT_ID,
                    CLIENT_SECRET)
line_bot_api = LineBotApi(YOUR_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(YOUR_CHANNEL_SECRET)
wikipedia.set_lang("ja")
#-----------------------------------------------------------------------------
# japanase to roman
def j2roma(src):
    kakasi = pykakasi.kakasi()
    kakasi.setMode('H', 'a')
    kakasi.setMode('K', 'a')
    kakasi.setMode('J', 'a')
    conv = kakasi.getConverter()
    return conv.do(src).replace("tsu","tu")

# roman to japanese
def roma2j(src):
    return romkan.to_katakana(src)

# transfer
def s2sh(src):
    dst = src.replace("sa","sha")
    dst = dst.replace("shi","shi")
    dst = dst.replace("su","shu")
    dst = dst.replace("se","she")
    dst = dst.replace("so","sho")
    dst = dst.replace("za", "ja")
    dst = dst.replace("zu", "ju")
    dst = dst.replace("zo", "jo")
    dst = dst.replace("tu", "tyu")
    dst = dst.replace("te", "che")
    dst = dst.replace("to", "cho")
    return dst , src != dst

# apply trasfer
def transfer(text):
    parse_li = coy.parse(text)
    compose_sentence = ""
    for parse in parse_li:
        for token in parse.tokens:
            if token.pos in ["括弧", "句点", "読点" ,"空白", "Symbol", "Number"]:
                compose_sentence += token.form
            else:
                result, is_diff = s2sh(j2roma(token.form))
                if is_diff:
                    compose_sentence += jaconv.kata2hira(roma2j(result))
                else:
                    compose_sentence += token.form
    return compose_sentence

#-----------------------------------------------------------------------------------------


@app.route("/")
def hello_world():
    return "Somehow you've successfully deployed the app"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# process text message
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    ai_message = talk_ai(event.message.text)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=ai_message))
def talk_ai(word):
    if "を調べて" in word:
        try:
            wikipedia_page = wikipedia.page(word.replace("を調べて", ""))
            wikipedia_title = wikipedia_page.title
            wikipedia_url = wikipedia_page.url
            wikipedia_summary = wikipedia.summary(word.replace("を調べて", ""))
            reply_message = "---" + wikipedia_title + "---" + "\n" + wikipedia_summary + "\n\n" + "URL:" + wikipedia_url
        except wikipedia.exceptions.PageError:
            reply_message = "---" + word.replace("を調べて", "") + "は見つかりませんでした。"
        except wikipedia.exceptions.DisambiguationError as e:
            disambiguation_list = e.options
            reply_message = "候補が複数あるので近い言葉を検索し直してください"
            for word in disambiguation_list:
                reply_message += word + "\n"
        return reply_message

    elif "を調べちぇ" in word:
        try:
            wikipedia_page = wikipedia.page(word.replace("を調べちぇ", ""))
            wikipedia_title = wikipedia_page.title
            wikipedia_url = wikipedia_page.url
            wikipedia_summary = wikipedia.summary(word.replace("を調べちぇ", ""))
            reply_message = "---" + transfer(wikipedia_title) + "---" + "\n" + transfer(wikipedia_summary) + "\n\n" + "URL:" + wikipedia_url
        except wikipedia.exceptions.PageError:
            reply_message = "---" + word.replace("を調べちぇ", "") + "は見ちゅかりましぇんでした。"
        except wikipedia.exceptions.DisambiguationError as e:
            disambiguation_list = e.options
            reply_message = "候補がふくしゅうあるので近いこちょばをけんしゃくし直しちぇくだしゃい"
            for word in disambiguation_list:
                reply_message += word + "\n"
        return reply_message
    
    elif "を変換しちぇ" in word:
        return transfer(word.replace("を変換しちぇ", ""))

    else:
        client = pya3rt.TalkClient(PYART_API_KEY)
        reply_message = client.talk(word)
        return reply_message["results"][0]["reply"]

# process Image message
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)

    if os.path.exists("./static"):
        shutil.rmtree("./static")
    if not os.path.isdir("./static"):
        os.mkdir("./static")

    with open("static/"+event.message.id+".jpg", "wb") as f:
        f.write(message_content.content)
        test_url = "./static/" + event.message.id+".jpg"
        img = tfms(Image.open(test_url)).unsqueeze(0)
        
    with torch.no_grad():
        outputs = net(img)

    result_list = []
    for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        result_list.append('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))
    message = "\n".join(result_list)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=message))


# run
if __name__ == "__main__":

    # load efficientnet and labels
    net = EfficientNet.from_pretrained("efficientnet-b0")
    net.eval()
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    labels_map = json.load(open("labels.txt"))
    labels_map = [labels_map[str(i)] for i in range(1000)]
    
    # run app
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
