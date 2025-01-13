import streamlit as st
import plotly.express as px
import pandas as pd
import requests
from PIL import Image
import io
import base64
import plotly.graph_objects as go



@st.experimental_memo
def get_chart_dist(response_dic):
    dist_player1 = response_dic['distance']["last_frame_distance_player1"]
    dist_player2 = response_dic['distance']["last_frame_distance_player2"]

    stages = ["Running Distance"]
    df_mtl = pd.DataFrame(dict(number=[dist_player1], stage=stages))
    df_mtl['Player'] = 'Player 1'
    df_toronto = pd.DataFrame(dict(number=[dist_player2], stage=stages))
    df_toronto['Player'] = 'Player 2'
    df = pd.concat([df_mtl, df_toronto], axis=0)
    fig = px.funnel(df, x='number', y='stage', color='Player')
    fig.update_layout(autosize=False, width=300)  # Set the width of the chart here

    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit")
    with tab2:
        st.plotly_chart(fig, theme=None)

@st.experimental_memo
def get_chart_strokes(response_dic):
    # Get stroke counts for Player 1
    services_player1 = response_dic.get('stroke_counts').get("Service/Smash")
    backhand_player1 = response_dic.get('stroke_counts').get("Backhand")
    forehand_player1 = response_dic.get('stroke_counts').get("Forehand")

    # Create DataFrame for Player 1 strokes
    strokes_data = {
        "Stroke Type": ["Serve/Smash", "Forehand", "Backhand"],
        "Player 1": [services_player1, forehand_player1, backhand_player1]
    }
    df = pd.DataFrame(strokes_data)

    # Plotting
    fig = px.bar(df, x='Stroke Type', y='Player 1', color='Stroke Type',
                 color_discrete_map={'Serve/Smash': 'lightblue', 'Forehand': 'blue', 'Backhand': 'darkblue'},
                 labels={'Player 1': 'Count'})

    # Customize layout
    fig.update_layout(autosize=False, width=500)  # Set the width of the chart here

    # Display chart in tabs
    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit")
    with tab2:
        st.plotly_chart(fig, theme=None)

@st.experimental_memo
def get_table_model_info(response_dic):
    fig = go.Figure(data=[go.Table(header=dict(values=['Model stats', 'Values']),
                                   cells=dict(values=[["Court detection time (s)", "Court detection accuracy (%)", 'Total computation time (s)', 'Total frames analyzed'],
                                                      [response_dic.get('court_detection_time'), response_dic.get('court_accuracy'), round(response_dic.get('Total computation time (s)'), 2), response_dic.get('total_frames_analyzed')]]))])
    fig.update_layout(autosize=False, width=300)  # Set the width of the table here

    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit")
    with tab2:
        st.plotly_chart(fig, theme=None)


'''
# Tennis Shot Recognition
'''

st.markdown('''
*The future of tennis*

Professional stats from your tennis video!
''')

'''
## Please download your video here

'''
# Initialize session state
if 'heatmap_img' not in st.session_state:
    st.session_state['heatmap_img'] = None
if 'graph_img' not in st.session_state:
    st.session_state['graph_img'] = None
if 'game_stats' not in st.session_state:
    st.session_state['game_stats'] = None

video_file = st.file_uploader("Choose a 🎾 mp4 video file ", type=["mp4", "avi", "mov", "mkv"])
if st.button("Send the video file", type="primary") and video_file is not None:
    files = {"file": video_file.getvalue()}
    res = requests.post('http://104.155.13.104:8000/savefile', files=files)
    if res.ok:
        response = res.json()
        video_data = response.get('video', '')
        if video_data:
            # Decode base64 encoded video data
            video_bytes = base64.b64decode(video_data)
            # Create a download button
            st.download_button(label="Download video",
                               data=video_bytes,
                               file_name='output_video.mp4',
                               mime='video/mp4')

        #get images data
        heatmap_data = response.get('heatmap', '')
        graph_data = response.get('graph', '')

        # Decode base64 encoded images data
        heatmap_bytes = base64.b64decode(heatmap_data)
        graph_bytes = base64.b64decode(graph_data)

        # Create images and store in session state
        st.session_state['heatmap_img'] = Image.open(io.BytesIO(heatmap_bytes))
        heatmap_img = heatmap_img.resize((heatmap_img.width // 2, heatmap_img.height // 2))
        st.session_state['graph_img'] = Image.open(io.BytesIO(graph_bytes))

        # Store game statistics in session state
        st.session_state['game_stats'] = response['result_json']


# Display images and game statistics from session state
def display_game_stats():
    if st.session_state['heatmap_img'] is not None and st.session_state['game_stats'] is not None:
        '''
        ## Game Statistics
        '''

        dist_player1 = st.session_state['game_stats']['distance']['last_frame_distance_player1']
        dist_player2 = st.session_state['game_stats']['distance']['last_frame_distance_player2']

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label='Player 1 Running Distance', value=f"{dist_player1} meters")
        with col2:
            st.metric(label='Player 2 Running Distance', value=f"{dist_player2} meters")


        st.write('Strokes sequence')
        st.write(st.session_state['game_stats']['stroke'])
        st.write('Strokes played by player 1')
        get_chart_strokes(st.session_state['game_stats'])

        st.image(st.session_state['heatmap_img'], use_column_width=True, width=300)
        pass

def display_model_performance():
    if st.session_state['heatmap_img'] is not None and st.session_state['game_stats'] is not None:
        '''
        ### Model performance
        '''
        get_table_model_info(st.session_state['game_stats'])
        pass


if st.session_state['heatmap_img'] is not None and st.session_state['game_stats'] is not None:
    # Navigation setup
    PAGES = {
        "Game Statistics": display_game_stats,
        "Model Performance": display_model_performance
    }

    # Streamlit UI
    st.sidebar.title("Navigation")
    if st.session_state['game_stats'] is not None:
        selection = st.sidebar.radio("Go to", list(PAGES.keys()))
        page = PAGES[selection]
        page()
    else:
        st.write("Please upload a video to see the game statistics and model performance.")

    re_run = False

#st.image(graph_img, caption="Players and ball movement on y axis", use_column_width=True)
