#!/usr/bin/env python
# coding: utf-8

# In[181]:


import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go


# In[182]:


df = pd.read_csv("usa_house.csv")
df2 = pd.read_csv("kc_house.csv")
df3 = pd.read_csv("API_USA_Housing.csv")


# In[183]:


df = df[df['price'] != 0]


# In[184]:


tab1, tab2, tab3 = st.tabs(["Data verkenning", "Analyse", "Vergelijking"])


# https://docs.streamlit.io/library/api-reference/charts/st.map
# 
# https://plotly.com/python/bubble-maps/

# **te zien hoeveel en waar de huizen zijn**
# 
# * op de kaart zelf te zien 
# * ene kleur is voor de ene df en andere kleur voor de andere
# * het bouwjaar kan verschoven worden om alleen huizen in die bepaalde range te laten zien

# In[185]:


with tab1:
    st.title("Verdeling van huizen in Washington")

    st.write(
    "In deze plot wordt de verdeling van de huizen weergegeven."
    " De lichtblauwe bolletjes vertegenwoordigen de data van de tweede dataframe,"
    " terwijl de donkerblauwe bolletjes de data van de eerste dataframe weergeven."
    " Op de kaart is te zien dat de huizen zich in de staat Washington bevinden."
    " Er zijn meerdere lichtblauwe bolletjes zichtbaar, omdat de data in deze dataframe gedetailleerder was."
    " Hierbij was de exacte locatie van de huizen bekend."
    " Voor de andere dataframe hebben we postcodes gebruikt om de locaties vast te leggen."
    " Hiervoor hebben we extra kolommen toegevoegd om de locatie te berekenen op basis van de postcodes."
    )


# In[186]:


with tab1:
    min_bouwjaar, max_bouwjaar = st.slider('Bouwjaren', 
                                          min_value=min(df['yr_built']), 
                                          max_value=max(df['yr_built']), 
                                          value=(min(df['yr_built']), max(df['yr_built'])))

    # Filter de gegevens op basis van het bouwjaar
    filtered_df = df[(df['yr_built'] >= min_bouwjaar) & (df['yr_built'] <= max_bouwjaar)]
    filtered_df2 = df2[(df2['yr_built'] >= min_bouwjaar) & (df2['yr_built'] <= max_bouwjaar)]

    fig = px.scatter_mapbox(filtered_df2, lat="lat", lon="long", 
                            color_discrete_sequence=["lightblue"])
    fig.add_trace(px.scatter_mapbox(filtered_df, lat="latitude", lon="longitude", 
                                    color_discrete_sequence=["steelblue"]).data[0])

    # Opmaak van de kaart
    fig.update_layout(
        mapbox_style="carto-positron",  
        autosize=False, 
        width = 700, 
        height = 500
    )
    
    # Toon de kaart
    # fig.show()

    st.plotly_chart(fig)


# **Gemiddelde huisprijzen per stad per jaar**
# 
# * drop down optie met stad
# * x-as is het jaar
# * y-as is de gemiddelde prijs

# In[187]:


with tab1:
    st.title("Gemiddelde Huisprijzen per Jaar")

    st.write(
        "In de volgende plot zijn de gemiddelde huisprijzen per jaar te zien."
        " Hierbij zijn beide dataframes in één plot samengevoegd."
        " Via de dropdown-box kan er een stad worden geselecteerd om naar te kijken."
        " Er is alleen gefilterd op steden die in beide dataframes voorkomen."
        " Voor de tweede dataframe is er een kolom toegevoegd om de locatie om te zetten naar de desbetreffende stad."
    )


# In[188]:


with tab1:

    # Dropdown-menu voor steden
    steden_df1 = set(df['city'])
    steden_df2 = set(df2['city'])
    gemeenschappelijke_steden = steden_df1.intersection(steden_df2)
    selected_city = st.selectbox('Selecteer een stad', gemeenschappelijke_steden)

    # Filter de dataset op de geselecteerde stad
    filtered_data = df[df['city'] == selected_city]
    filtered_data2 = df2[df2['city'] == selected_city]


    # Groepeer de data per jaar en stad en bereken het gemiddelde per jaar
    avg_price_per_year = filtered_data.groupby(['yr_built', 'city'])['price'].mean().reset_index()
    avg_price_per_year2 = filtered_data2.groupby(['yr_built', 'city'])['price'].mean().reset_index()


    # Maak de Plotly-lijngrafiek
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=avg_price_per_year["yr_built"], y=avg_price_per_year["price"], mode='lines', name='Dataframe 1', line=dict(color='lightblue')))
    fig.add_trace(go.Scatter(x=avg_price_per_year2["yr_built"], y=avg_price_per_year2["price"], mode='lines', name='Dataframe 2', line=dict(color='steelblue')))

    fig.update_layout(
        title=f"Prijsontwikkeling in {selected_city}",
        xaxis_title="Jaar",
        yaxis_title="Prijs",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig)


# In[189]:


with tab1:

    st.title("Prijs versus Bouwjaar van Huizen")

    st.write(
        "In de volgende plot is de prijs tegenover het bouwjaar van huizen geplaatst."
        " Er is onderscheid gemaakt tussen gerenoveerde en niet-gerenoveerde huizen."
        " Een checkbox is toegevoegd waarmee je de plot met of zonder uitschieters (uitbijters) kunt bekijken."
        " In de legenda wordt aangegeven welke kleur wordt gebruikt voor gerenoveerde en niet-gerenoveerde huizen."
        " Met de dropdown-box kun je de gewenste dataset kiezen."
    )


# In[190]:


with tab1:
    #verwijderen van uitbijters
    def remove_outliers(df, threshold=2):
        z_scores = np.abs((df - df.mean()) / df.std())
        return df[(z_scores < threshold).all(axis=1)]

    # Streamlit-applicatie
    dataset_keuze = st.selectbox('Selecteer een dataset', ['Dataset 1 USA', 'Dataset 2 King County'], key = 'datasetkeuze1')

    # Voeg een dropdown-menu toe voor het filteren van uitbijters
    show_outliers = st.checkbox('Toon uitbijters', key='show_outliers', value=False)

    # Filter de data op basis van de geselecteerde optie
    if show_outliers:
        filtered_data = df[['yr_built', 'price', 'yr_renovated']]
        filtered_data2 = df2[['yr_built', 'price', 'yr_renovated']]
    else:
        filtered_data = remove_outliers(df[['yr_built', 'price', 'yr_renovated']])
        filtered_data2 = remove_outliers(df2[['yr_built', 'price', 'yr_renovated']], 5)

    # Maak een nieuwe kolom om het renovatiestatus weer te geven
    filtered_data['Renovatie_Status'] = filtered_data['yr_renovated'].apply(lambda x: 'Gerennoveerd' if x != 0 else 'Niet Gerennoveerd')
    filtered_data2['Renovatie_Status'] = filtered_data2['yr_renovated'].apply(lambda x: 'Gerennoveerd' if x != 0 else 'Niet Gerennoveerd')

    #colorpaltete
    color_palet = {'Gerennoveerd':'lightblue', 'Niet Gerennoveerd':'steelblue'}

    # Scatter plot met Plotly Express
    if dataset_keuze == 'Dataset 1 USA':
        fig = px.scatter(filtered_data, x='yr_built', y='price', color='Renovatie_Status', color_discrete_map=color_palet, title='Scatter Plot van Prijs vs. Bouwjaar')
    else:
        fig = px.scatter(filtered_data2, x='yr_built', y='price', color='Renovatie_Status', color_discrete_map=color_palet, title='Scatter Plot van Prijs vs. Bouwjaar')
        fig.update_layout(legend=dict(traceorder = 'reversed'))
    fig.update_xaxes(title_text='Bouwjaar')
    fig.update_yaxes(title_text='Prijs')

    # Toon de plot in Streamlit
    st.plotly_chart(fig)


# In[191]:


with tab2:
    df['numberofrooms'] = df['bedrooms'] + df['bathrooms'] + 1
    df2['numberofrooms'] = df2['bedrooms'] + df2['bathrooms'] + 1
    df = df[df['price'] != 0]
    df2 = df2[df2['price'] != 0]


# In[192]:


with tab2:
    # Titel
    st.title("Correlatietabel voor 'Price' vs Andere Variabelen")

    # Inleidingstekst
    st.write(
        "In de onderstaande tabel is de correlatie weergeven van price met de andere variabelen uit de dataset. "
        "Voordat de correlatie wordt berekend, zijn de uitbijters verwijderd door een ondergrens en bovengrens te bepalen "
        "met behulp van de kwartielafstand. Door middel van de knopen kan er gewisseld worden van dataset."
    )

    # Alinea over correlatie met sqft_living
    st.write(
        "In beide datasets is te zien dat 'sqft_living' de grootste correlatie heeft in vergelijking met de andere variabelen. "
        "Daarom wordt er met deze variabele een voorspelling gemaakt van de price aan de hand van lineaire regressie."
    )


# In[193]:


with tab2:
    
    def verwijder_uitbijters(df, var):
        Q3 = df[var].quantile(0.75)
        Q1 = df[var].quantile(0.25)
        IQR = Q3-Q1
        upper = Q3 + (1.5 * IQR)
        lower = Q1 - (1.5 * IQR)

        df = df[(df[var] > lower) & (df[var] < upper)]

        return df

    df_cor = df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                'condition', 'sqft_above', 'sqft_basement']]
    df_cor2 = df2[['price', 'bedrooms', 'bathrooms', 'sqft_living',
               'sqft_lot', 'floors', 'condition', 'grade',
               'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']]

    # Dropdown-menu voor weergave met of zonder uitbijters
    show_outliers33 = st.checkbox('Toon uitbijters', key='show_outliers33', value=False)
    dataset = st.selectbox('Selecteer een dataset', ['Dataset 1 USA', 'Dataset 2 King County'], key = 'datasetkeuze2')


    # Verwijder uitbijters op basis van de geselecteerde kolom en weergaveoptie
    if show_outliers33:
        df_cor = df_cor
        df_cor2 = df_cor2
    else:
        # Lijst van kolommen om uitbijters te verwijderen
        columns_to_process = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                'condition', 'sqft_above', 'sqft_basement']

        #Loop door de kolommen en verwijder uitbijters, bijwerkend DataFrame 'df'
        for column in columns_to_process:
            df_cor = verwijder_uitbijters(df_cor, column)

        # Lijst van kolommen om uitbijters te verwijderen in df2
        columns_to_process_df2 = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
               'sqft_lot', 'floors', 'condition', 'grade',
               'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']

        # Loop door de kolommen en verwijder uitbijters, bijwerkend DataFrame 'df2'
        for column in columns_to_process_df2:
            df_cor2 = verwijder_uitbijters(df_cor2, column)


    # Kies de kolom waarvoor je de correlatie wilt berekenen
    column_to_correlate = 'price'

    # Bereken de correlatiecoëfficiënten tussen de geselecteerde kolom en andere numerieke kolommen
    correlation_matrix = df_cor.corr()
    correlation_matrix2 = df_cor2.corr()

    # Kies alleen de correlatie van de geselecteerde kolom met andere kolommen
    correlation_with_target = correlation_matrix[column_to_correlate]
    correlation_with_target2 = correlation_matrix2[column_to_correlate]
    # Maak een bar plot om de correlatie te visualiseren
    fig1 = px.bar(
        x=correlation_with_target.index,
        y=correlation_with_target.values,
        labels={'x': 'Kolom', 'y': f'Correlatie met {column_to_correlate}'},
        title=f'Correlatie met {column_to_correlate}',
        color=correlation_with_target.values
    )
    fig1.update_xaxes(tickangle=90)

    # Maak een bar plot om de correlatie te visualiseren
    fig2 = px.bar(
        x=correlation_with_target2.index,
        y=correlation_with_target2.values,
        labels={'x': 'Kolom', 'y': f'Correlatie met {column_to_correlate}'},
        title=f'Correlatie met {column_to_correlate}',
        color=correlation_with_target2.values
    )
    fig2.update_xaxes(tickangle=90)

    fig1_button = {'method': 'update', 'label': 'Figure 1', 'args': [{'visible': [True, False]}, {'title': 'Figure 1'}]}
    fig2_button = {'method': 'update', 'label': 'Figure 2', 'args': [{'visible': [False, True]}, {'title': 'Figure 2'}]}

#     # Voeg knoppen toe voor de figuren
#     button_1 = st.button('Data 1')
#     button_2 = st.button('Data 2')


    if dataset == 'Dataset 1 USA':
        st.plotly_chart(fig1)
    elif dataset == 'Dataset 2 King County':
        st.plotly_chart(fig2)


# In[194]:


with tab2:

    # Titel
    st.title("Regressie")

    # Inleidingstekst
    st.write(
        "In de onderstaande grafiek is een scatterplot weergegeven met een regressielijn. "
        "Door middel van het dropdownmenu kan er een variabele geselecteerd worden om te vergelijken met de variabele 'price'. "
        "Door gebruik te maken van de knoppen kan er gewisseld worden tussen datasets en kun je de regressielijnen vergelijken tussen de verschillende datasets. "
        "Tenslotte kun je ook vergelijken of de uitbijters van invloed zijn op de regressie door uitbijters aan of uit te zetten."
    )

    # Alinea over R2-score en sqft_living
    st.write(
        "In beide datasets is te zien dat de variabele 'sqft_living' de beste R2-score geeft. "
        "Dit was te verwachten aangezien deze variabele ook de grootste correlatie had met 'price'. "
        "Echter valt wel op dat als je uitbijters weghaalt bij dataset 1, de R2-score verbetert. "
        "Maar dit is niet het geval bij dataset 2; als je hier de uitbijters weghaalt, wordt de R2-score slechter."
    )

    # Voeg hier je dropdownmenu, knoppen en andere interactieve elementen toe


# In[195]:


with tab2:

    def verwijder_uitbijters(df, var):
        Q3 = df[var].quantile(0.75)
        Q1 = df[var].quantile(0.25)
        IQR = Q3-Q1
        upper = Q3 + (1.5 * IQR)
        lower = Q1 - (1.5 * IQR)

        df = df[(df[var] > lower) & (df[var] < upper)]

        return df

    df_sqft = df[['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']]
    df_sqft2 = df2[['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']]

#     df = df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
#                 'condition', 'sqft_above', 'sqft_basement']]
#     df2 = df2[['price', 'bedrooms', 'bathrooms', 'sqft_living',
#                'sqft_lot', 'floors', 'condition', 'grade',
#                'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']]

    # Dropdown-menu voor kenmerken op de x-as
    selected_feature = st.selectbox('Selecteer een kenmerk voor de x-as', df_sqft.columns)
    dataset2 = st.selectbox('Selecteer een dataset', ['Dataset 1 USA', 'Dataset 2 King County'], key = 'datasetkeuze3')


    # Dropdown-menu voor weergave met of zonder uitbijters
    show_outliers666 = st.checkbox('Toon uitbijters', key='show_outliers666', value=False)

    # Verwijder uitbijters op basis van de geselecteerde kolom en weergaveoptie
    if show_outliers666:
        df = df
        df2 = df2
    else:
        # Lijst van kolommen om uitbijters te verwijderen
        columns_to_process = ['price', selected_feature]

        # Loop door de kolommen en verwijder uitbijters, bijwerkend DataFrame 'df'
        for column in columns_to_process:
            df = verwijder_uitbijters(df, column)

        # Lijst van kolommen om uitbijters te verwijderen in df2
        columns_to_process_df2 = ['price', selected_feature]

        # Loop door de kolommen en verwijder uitbijters, bijwerkend DataFrame 'df2'
        for column in columns_to_process_df2:
            df2 = verwijder_uitbijters(df2, column)


    # Maak een scatterplot met OverallQual op de x-as en SalePrice op de y-as
    fig1 = px.scatter(df, x=selected_feature, y='price')
    fig2 = px.scatter(df2, x=selected_feature, y='price')

    fig1.update_traces(marker=dict(color='steelblue'))
    fig2.update_traces(marker=dict(color='lightblue'))

    # Stel de kleur van de trendlijn in op rood (R)
    fig1.update_traces(line=dict(color='red'))
    fig2.update_traces(line=dict(color='red'))

    fig1_button = {'method': 'update', 'label': 'Figure 1', 'args': [{'visible': [True, False]}, {'title': 'Figure 1'}]}
    fig2_button = {'method': 'update', 'label': 'Figure 2', 'args': [{'visible': [False, True]}, {'title': 'Figure 2'}]}

#     # Voeg knoppen toe voor de figuren
#     button_3 = st.button('Dataset 1')
#     button_4 = st.button('Dataset 2')


    if dataset2 == 'Dataset 1 USA':
        st.plotly_chart(fig1)
    elif dataset2 == 'Dataset 2 King County':
        st.plotly_chart(fig2)


# In[196]:


with tab3:
    df['street'] = df['street'].str.upper()

    new_df = df[df['street'].isin(df3['address.line1'])]
    new_df3 = df3[df3['address.line1'].isin(df['street'])]


# In[197]:


with tab3:  

    # Voeg een nieuwe kolom 'Category' toe om de gegevens van beide DataFrames te onderscheiden
    new_df3['Category'] = 'Address Line 1'
    new_df['Category'] = 'Street'

    # Combineer de gegevens van beide DataFrames
    combined_data = pd.concat([new_df3, new_df])

    # Maak een staafdiagram waarbij de balken afwisselend worden weergegeven
    fig = go.Figure()

    categories = combined_data['Category'].unique()

    for category in categories:
        data_category = combined_data[combined_data['Category'] == category]
    if category == 'Address Line 1':
        marker_color = 'lightblue'
        year = '2014'
    else:
        marker_color = 'steelblue'
        year = '2022'
    fig.add_trace(go.Bar(x=data_category['address.line1'] if category == 'Address Line 1' else data_category['street'],
                         y=data_category['avm.amount.value'] if category == 'Address Line 1' else data_category['price'],
                         name=year,
                         marker_color=marker_color))

    # Aanpassen van de lay-out
    fig.update_layout(
        xaxis_title='Address Line 1 / Street',
        yaxis_title='Gemiddelde waarde',
        title='Vergelijking tussen Address Line 1 en Street'
    )

    # Toon de figuur
    st.plotly_chart(fig)


# In[ ]:





# In[ ]:




