import streamlit as st
import pandas as pd
import altair as alt



def Performance_matrix():


    st.title("Performance Matrix")
    
    data = {
            'Model': ['Google Plam', 'Mistal AI', 'Hugging_face API '],
            'PDF Chat': [23, 34, 23],
            'Summarization_time': [45, 56, 45],
            'Keyword Extraction': [66, 34, 34]
                }

        # Create DataFrame
    df = pd.DataFrame(data)


    # Create Altair Chart
    chart = alt.Chart(df).mark_bar().encode(
        x='Model',
        y='Summarization_time',
        color=alt.value('blue')  # Set the bar color


    )

    # Display the chart
    st.altair_chart(chart, use_container_width=True)

    chart2 = alt.Chart(df).mark_bar().encode(
        x='Model',
        y='Keyword Extraction',
        color=alt.value('red')  # Set the bar color

        
    )
    st.altair_chart(chart2, use_container_width=True)

    chart3 = alt.Chart(df).mark_bar().encode(
        x='Model',
        y='PDF Chat',
        color=alt.value()  # Set the bar color

        
    )
    st.altair_chart(chart3, use_container_width=True)



if __name__=="__main__":
    Performance_matrix()