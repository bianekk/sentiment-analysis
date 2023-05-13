import praw
from flask import Flask, render_template
import os
import statsd
import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
import altair as alt
import numpy as np
import timeit


# app = Flask(__name__)
stats = statsd.StatsClient('graphite', 8125)
reddit = praw.Reddit(
    client_id="Qt46AYEyeGlLLh7KVep4xw",
    client_secret="K7b0ii9VXAPFWEq-lIfiASS70hDVbQ",
    password="froggyH@t123",
    user_agent="7_pdiow_7",
    username="bianekk",
)

classifier = pipeline("text-classification", model='bhadresh-savani/albert-base-v2-emotion', return_all_scores=True)
# set page
st.set_page_config(page_title="pdiow(bianek)", layout="wide")
st.title("Sentiment analysis app")
input = st.text_input("Pass a subreddit to check", value='meme')
apply_button = st.button("Apply")
labels = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]


@stats.timer('reddit_scrap.request_times')
@st.cache_data
def get_subreddit(input_val):
    stats.incr('reddit_scrap.requests')
    headlines = []
    start = timeit.default_timer()
    for submission in reddit.subreddit(input_val).hot(limit=None):
        element = {'title': submission.title, 'date': submission.created_utc,
                   'url': submission.url, 'score': submission.score,
                   'upvote_ration': submission.upvote_ratio
                   }
        headlines.append(element)
    end = timeit.default_timer()
    time = (end - start)
    stats.timing('reddit_scrap.time', time)
    df = pd.DataFrame(headlines)
    df['date'] = pd.to_datetime(df['date'], unit='s')
    return df


def get_main_emotion(text):
    result = classifier(text)
    sorted_result = sorted(result[0], key=lambda x: x['score'], reverse=True)
    emotion = sorted_result[0]['label']
    return emotion


def handle_subreddit():
    if input:
        with st.spinner('Analysis in progress'):
            progress_bar = st.progress(0)
            for row_num, row in df.iterrows():
                df.at[row_num, 'main_emotion'] = get_main_emotion(row['title'])
                progress_percent = int((row_num + 1) / len(df) * 100)
                progress_bar.progress(progress_percent)


if apply_button:
    try:
        df = get_subreddit(input)
        handle_subreddit()

        st.write("Data")
        st.dataframe(df)

        # posts over time
        fig = px.histogram(df['date'], x="date", title='Number of posts over time')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, color="#e08093")

        chart_emotion_total = alt.Chart(df).mark_bar().encode(
            x='count()',
            y='main_emotion',
            color=alt.value('#e08093')

        )
        # overall sentiment
        st.write("Overall sentiment")
        st.altair_chart(chart_emotion_total, use_container_width=True)
        fig2 = fig = px.histogram(df, x="date", color='main_emotion', title='Overall sentiment over time')
        st.plotly_chart(fig2, theme="streamlit", use_container_width=True, color="#e08093")

        # emotions over time
        percent_emotions = df['main_emotion'].value_counts(normalize=True) * 100

        emotion_dict = df['main_emotion'].value_counts().to_dict()
        sum_all = sum(emotion_dict.values())
        formated_dict = {"emotion": [el for el in emotion_dict],
                         "value": [val / sum_all for key, val in emotion_dict.items()]}
        emotion_df = pd.DataFrame.from_dict(formated_dict)
        st.write("Percentage share of emotions in all posts")
        pie = alt.Chart(emotion_df).mark_arc().encode(
            theta="value",
            color="emotion"
        )
        st.altair_chart(pie, use_container_width=True)

        # upscore ratio vs emotions
        st.write("Emotions over upvote ratio")
        sadness = df.loc[df['main_emotion'] == 'sadness']
        sadness = np.mean(sadness['upvote_ration'])
        joy = df.loc[df['main_emotion'] == 'joy']
        joy = np.mean(joy['upvote_ration'])
        love = df.loc[df['main_emotion'] == 'love']
        love = np.mean(love['upvote_ration'])
        anger = df.loc[df['main_emotion'] == 'anger']
        anger = np.mean(anger['upvote_ration'])
        fear = df.loc[df['main_emotion'] == 'fear']
        fear = np.mean(fear['upvote_ration'])
        surprise = df.loc[df['main_emotion'] == 'surprise']
        surprise = np.mean(surprise['upvote_ration'])
        emotion = [sadness, joy, love, anger, fear, surprise]
        df_res = pd.DataFrame(list(zip(emotion, labels)), columns=['upvote_ratio', 'emotion'])
        # st.dataframe(df_res)
        chart_emotion_score = alt.Chart(df_res).mark_bar().encode(
            x='emotion',
            y='upvote_ratio',
            color=alt.value('#e08093')
        )
        st.altair_chart(chart_emotion_score, use_container_width=True)

        df_res = df[['date', 'main_emotion']]
        df_res['date_utc'] = df_res['date'].dt.tz_localize('UTC')

        chart_emotion_time = alt.Chart(df_res).mark_circle(size=60).encode(
            x='main_emotion',
            y='utchoursminutes(date_utc)',
            color='main_emotion'
        )
        st.altair_chart(chart_emotion_time, use_container_width=True)
    except:
        st.warning('Subreddit failed to load, perhaps a typo?')
