import streamlit as st
#from public_space import *


st.set_page_config(
    page_title="Urban Modeller",
    page_icon="./sl//favicon.ico",
    layout='wide',
    initial_sidebar_state='collapsed')

st.write(
        """
<iframe src="resources/sidebar-closer.html" height=0 width=0>
</iframe>""",
        unsafe_allow_html=True,
    )

# CSS
with open('./sl/style.css') as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


menu_list = st.sidebar.radio('Navigation', ["Home", "Public space", "Mobility"])

if menu_list == "Home":
    st.write("LANDING DE LA PAGINA")

elif menu_list=="Public space":
    st.write("Hacer las tres visualizaciones aca")