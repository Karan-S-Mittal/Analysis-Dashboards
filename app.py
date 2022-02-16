import nltk
import pandas as pd

nltk.download("averaged_perceptron_tagger")
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import scattertext as st
import texthero as hero
from dash import dcc, html
from dash.dash_table import DataTable
from dash.dependencies import Input, Output, State

from utils import get_page_source, make_wordcloud


external_stylesheets = [
    dbc.themes.BOOTSTRAP,
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Data Scientist Job Analysis"

# read data
data_scientist = pd.read_csv("./data/data-science-job-search.csv")
data_engineer = pd.read_csv("./data/data-engineer-job-search.csv")
data_scientist["title"] = ["data scientist"] * len(data_scientist)
data_engineer["title"] = ["data engineer"] * len(
    data_scientist
)  # adding data scientist with keeping the column same.
df = pd.concat([data_scientist, data_engineer]).reset_index(
    drop=True
)  # join the two datasets
df = df[~df.text.isna()]  # remove null values
df["page_host"] = df["pageUrl"].apply(get_page_source)

# analysing top 20 pages
top_pages = df["page_host"].value_counts()[:20].to_frame()

top_20_fig = px.histogram(
    top_pages,
    x=top_pages.index,
    y="page_host",
    labels={"sum of page_host": "frequency", "index": "page host"},
).update_xaxes(
    categoryorder="total descending",
)

# Clean text
df["text"] = df["text"].pipe(hero.clean)

# Turn a list of text into a string
text = " ".join(df["text"].values)

# make wordcloud
text_cloud = make_wordcloud(text)
analyze_col = "requirements"

# Filter out the rows whose requirement is nan
filtered_df = df[~df[analyze_col].isna()][["title", analyze_col, "page_host"]]

# Tokenize text
filtered_df["parse"] = filtered_df[analyze_col].apply(st.whitespace_nlp_with_sentences)

corpus = (
    st.CorpusFromParsedDocuments(filtered_df, category_col="title", parsed_col="parse")
    .build()
    .get_unigram_corpus()
    .compact(st.AssociationCompactor(2000))
)

# get DataFrame with terms and their frequency
term_freq_df = corpus.get_term_freq_df()

# Get scaled F-scores of each term in each category
term_freq_df["Data Scientist Score"] = corpus.get_scaled_f_scores("data scientist")
term_freq_df["Data Engineer Score"] = corpus.get_scaled_f_scores("data engineer")

# Remove terms that are not nouns
def is_noun(word: str):
    pos = nltk.pos_tag([word])[0][1]
    return pos[:2] == "NN"


term_freq_df = term_freq_df.loc[map(is_noun, term_freq_df.index)]
# Get terms with the highest data scientist F-scores:
data_science_score = term_freq_df.sort_values(
    by="Data Scientist Score", ascending=False
).index[:30]
# get terms with highest data engineer F-scores:
data_engineer_score = term_freq_df.sort_values(
    by="Data Engineer Score", ascending=False
).index[:30]

requirement_terms = st.produce_scattertext_explorer(
    corpus,
    category="data scientist",
    category_name="Data scientist",
    not_category_name="Data Engineer",
    minimum_term_frequency=5,
    pmi_threshold_coefficient=0,
    width_in_pixels=1000,
    metadata=corpus.get_df()["page_host"],
    transform=st.Scalers.dense_rank,
)
# open("data_science_vs_data_engineer_requirements_terms.html", "w").write(html)

# Application Layout
app.layout = dbc.Container(
    [
        # graphs
        dbc.Row(
            [
                # top 20 pages
                dbc.Col([dcc.Graph(figure=top_20_fig)], width=6),
                # word cloud
                dbc.Col([html.Img(src=text_cloud, alt="wordcloud")], width=6),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Iframe(
                        title="Requirement Terms Analysis",
                        srcDoc=requirement_terms,
                        src="./data_science_vs_data_engineer_requirements_terms.html",
                        style={"height": "700px", "width": "100%"},
                    ),
                    width=True,
                ),
            ],
        ),
        # datatable
        dbc.Row(
            [
                # dataframe
                dbc.Col(
                    DataTable(
                        id="main_data",
                        columns=[{"id": i, "name": i} for i in df.columns],
                        data=df.to_dict("records"),
                        page_size=5,
                        style_cell={
                            "textOverflow": "ellipsis",
                            "whiteSpace": "normal",
                            "height": "auto",
                            "lineHeight": "15px",
                            "textAlign": "left",
                            "minWidth": "180px",
                            "width": "180px",
                            "maxWidth": "180px",
                        },
                        style_table={"overflowX": "auto"},
                        # conditionl_style_header = {
                        #     "if":
                        # }
                    )
                )
            ]
        ),
    ]
)

# running the application
if __name__ == "__main__":
    app.run_server(debug=True)
