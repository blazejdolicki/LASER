import xml.etree.ElementTree as ET

def get_label(label):
    if label=="4.0" or label=="5.0":
        return 1
    elif label=="1.0" or label=="2.0":
        return 0
    else:
        raise Exception("Invalid label")

text_index = {"de":5, "en":2, "fr":4,"jp":4}

langs = ['de','en','fr','jp']
category = 'books'
DATA_DIR = 'data/cls-acl10-unprocessed'

for lang in langs:
    for part in ['train','test']:
        FILE_DIR = f"{DATA_DIR}/{lang}/{category}/{part}"
        print(f'Creating file {FILE_DIR}.txt')
        tree = ET.parse(f'{FILE_DIR}.review')
        root = tree.getroot()
        with open(f"{FILE_DIR}.txt","w",encoding='utf-8') as f:
            for review in root:
                label = get_label(review[1].text)
                try:
                    text = review[text_index[lang]].text.replace("\n","")
                except:
                    # one of the Japanese reviews is empty setting it to "" was causing errors, so instead we take text from the title
                    text = review[text_index[lang]+1].text.replace("\n","")
                    print("No text in <text>")
                    print("Alternative text:",text)
                f.write("{}\t{}\n".format(label,text))