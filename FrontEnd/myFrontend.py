import requests
import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
 
# Title

st.set_page_config(page_title="ReelMe", page_icon="🍿", layout="centered", initial_sidebar_state="collapsed")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/private_files/lf30_bb9bkg1h.json")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')



with st.container():
    st.title("🍿ReelMe")
    st.subheader("About Us")
    st.markdown("**Welcome to ReelMe! This web app allows you to enter you and your friends favorite movies and reccomend one you will all love. No matter how diverse your tastes.✌️**")
    left_column,right_column = st.columns(2)
    with left_column:
        st.subheader("How to use ReelMe")
        st.text("")
        st.markdown("1. Enter your favorite movies.\n2. Click the button to get your reccomendation.\n3. Enjoy your movie night!")
    with right_column:
        # lotte file images
        st_lottie(lottie_coding, speed=.5, height=200, key="initial")
    if st.button("I am ready!"):        
        # move to the next page
        st.write("Let's get started!")

st.divider()

with st.container():
    friends = st.slider("How many people are you watching with?", 2, 10, 2)
    st.write(friends)
    my_list = []
    left_column, right_column = st.columns(2)

    for i in range(friends):
        if i%2 == 0:
            with left_column:
                my_list.append(st.text_input("Movie " + str(i+1), "Enter a movie..."))
        else:
            with right_column:
                my_list.append(st.text_input("Movie " + str(i+1), "Enter a movie..."))

    movieRecomendation = ""
    if st.button("Get your reccomendation"):
        st.subheader("You should watch: " + movieRecomendation)

# call calculations here with the passed in movies 



