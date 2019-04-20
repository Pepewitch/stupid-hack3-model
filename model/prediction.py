import pandas as pd
from os.path import join, dirname

baseline = 0.4
loveKeyword = {i.strip() for i in open(join(dirname(__file__),'love.txt'))}
hateKeyword = {i.strip() for i in open(join(dirname(__file__),'hate.txt'))}
reply_time_prop = 0.2
right_side_weight_prop = 0.2

def matchKeyword(keywords, message):
    n = 0
    for key in keywords:
        n += message.count(key)
    return n

def preprocess(df):
    df['datetime'] = pd.to_datetime(df['date']+' '+df['time'])
    df['love'] = df['message'].apply(lambda x: matchKeyword(loveKeyword, x))
    df['hate'] = df['message'].apply(lambda x: matchKeyword(hateKeyword, x))

def getRightSideWeight(df, owner):
    return len(df[df['name'] == owner])/len(df)

def getAverageReplyTime(df):
    reply = df[['name', 'datetime']]
    currentReply = reply['name'][0]
    currentTime = reply['datetime'][0]
    times = 0
    time = 0
    for index, row in reply.iterrows():
        if currentReply != row['name']:
            timedelta = (row['datetime'] - currentTime).value/60e9
            # assume 24 hrs is end of talking topic
            if timedelta < 24*60:
                time += timedelta
                times += 1
        currentTime = row['datetime']
        currentReply = row['name']
    return time / times

def predict(filepath, owner):
    df = pd.read_csv(filepath)
    preprocess(df)

    good = baseline
    
    right_side_weight = getRightSideWeight(df, owner)
    good += 2 * (0.5-right_side_weight) * right_side_weight_prop

    reply_time = getAverageReplyTime(df)
    if reply_time > 5:
        good -= min((reply_time-5)/20, reply_time_prop)
    else:
        good += min((5-reply_time)/20 ,reply_time_prop)
        
    
    return 0 if good < 0 else 1 if good > 1 else good


