import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
from dataset import show_Dataset_info
from EvaluateEmotionDetection import Show_My_Model
from ReportGen import Show_Report
from homepage import show_homepage
from UserTesting import Start_Testing




st.set_page_config(page_title="Emotion Detection", page_icon="mask", layout="wide")


sysmenu = '''
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
'''
st.markdown(sysmenu,unsafe_allow_html=True)

#css link
with open("style.css") as source_des:
	st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)
#end css link


with st.sidebar:
    choose = option_menu("", ["Homepage", "User Testing", "Dataset", "Experimentation", "Report"],
                        icons=['house-fill', 'person-bounding-box', 'card-text', 'eyedropper','journal-text'],
                        menu_icon="none", default_index=0,
                        styles={                                     
        "container": {"padding": "5!important", "background-color": "#00476D;", "border-radius":"0"},
        "icon": {"color": "#e9eef2", "font-size": "24px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#38688b","color":"#e9eef2","margin-bottom":"5px !important","border-radius":".3rem","padding":".4rem .5rem"},
        "nav-link-selected": {"background-color": "transparent","color":"#e9eef2;","padding":".4rem .5rem","border":"1px solid #DEF1F8"},
        
    }
    )

if choose == "Homepage":
    show_homepage()

elif choose == "User Testing":
    # Show_User_Testing()
    Start_Testing()

elif choose == "Dataset":
    show_Dataset_info()

elif choose == "Experimentation":
    Show_My_Model()


elif choose == "Report":
    Show_Report()


