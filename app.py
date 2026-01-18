"""
Synthetic Customer Persona & Market Research Tool
A Streamlit application for insurance customer segmentation and persona-based market research.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json
from typing import Dict, List, Tuple

# Page configuration
st.set_page_config(
    page_title="Synthetic Persona & Market Research Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4788;
        text-align: center;
        margin-bottom: 2rem;
    }
    .persona-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #1f4788;
    }
    .metric-card {
        background-color: #e3f2fd;
        border-radius: 8px;
        padding: 15px;
        margin: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f4788;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'personas' not in st.session_state:
    st.session_state.personas = None
if 'cluster_model' not in st.session_state:
    st.session_state.cluster_model = None
if 'cluster_stats' not in st.session_state:
    st.session_state.cluster_stats = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None


def initialize_openai_client(api_key: str) -> OpenAI:
    """Initialize OpenAI client with the provided API key."""
    return OpenAI(api_key=api_key)


def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, ColumnTransformer]:
    """
    Preprocess the customer data for clustering.
    Handles missing values, encodes categorical variables, and normalizes numerical features.

    Args:
        df: Raw customer dataframe

    Returns:
        Tuple of (scaled_features, cleaned_df, preprocessor)
    """
    # Make a copy to avoid modifying original
    df_clean = df.copy()

    # Define column types
    numerical_cols = []
    categorical_cols = []

    # Identify columns dynamically
    for col in df_clean.columns:
        col_lower = col.lower()
        if 'age' in col_lower or 'amount' in col_lower or 'year' in col_lower:
            numerical_cols.append(col)
        elif 'gender' in col_lower or 'city' in col_lower or 'brand' in col_lower or 'model' in col_lower:
            categorical_cols.append(col)

    # Create preprocessing pipelines
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    # Handle missing values
    for col in numerical_cols:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)

    for col in categorical_cols:
        df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)

    # Fit and transform
    X_scaled = preprocessor.fit_transform(df_clean)

    return X_scaled, df_clean, preprocessor


def perform_clustering(X: np.ndarray, n_clusters: int = 5) -> KMeans:
    """
    Perform K-Means clustering on preprocessed data.

    Args:
        X: Scaled feature matrix
        n_clusters: Number of clusters to create

    Returns:
        Fitted KMeans model
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    return kmeans


def calculate_cluster_statistics(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Calculate summary statistics for each cluster.

    Args:
        df: Cleaned customer dataframe
        labels: Cluster labels for each customer

    Returns:
        DataFrame with cluster statistics
    """
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = labels

    # Calculate statistics
    cluster_stats = []

    for cluster_id in sorted(df_with_clusters['Cluster'].unique()):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]

        stats = {
            'Cluster': f"Cluster {cluster_id + 1}",
            'Size': len(cluster_data),
            'Percentage': f"{len(cluster_data) / len(df_with_clusters) * 100:.1f}%"
        }

        # Add numerical statistics
        for col in cluster_data.select_dtypes(include=[np.number]).columns:
            if col != 'Cluster':
                stats[f'Avg_{col}'] = cluster_data[col].mean()

        # Add categorical mode
        for col in cluster_data.select_dtypes(include=['object']).columns:
            stats[f'Most_Common_{col}'] = cluster_data[col].mode()[0] if not cluster_data[col].mode().empty else 'N/A'

        cluster_stats.append(stats)

    return pd.DataFrame(cluster_stats)


def generate_persona_prompt(cluster_stats: Dict) -> str:
    """
    Create a prompt for the LLM to generate a persona based on cluster statistics.

    Args:
        cluster_stats: Dictionary containing cluster statistics

    Returns:
        Formatted prompt string
    """
    prompt = f"""Je bent een expert in customer persona ontwikkeling voor de Nederlandse verzekeringsmarkt.

Gebaseerd op de volgende klantdata, creëer een gedetailleerde, realistische buyer persona:

CLUSTER DATA:
{json.dumps(cluster_stats, indent=2)}

Creëer een persona met:
1. Een Nederlandse naam (voornaam en achternaam)
2. Leeftijd (afgerond)
3. Beroep/functie
4. Korte bio (2-3 zinnen over hun leven, gezinssituatie, en dagelijkse bezigheden)
5. Houding ten opzichte van verzekeringen en risico (1 paragraaf)
6. Een kenmerkende quote die hun mindset weergeeft
7. Belangrijkste waarden en prioriteiten
8. Grootste zorgen en angsten met betrekking tot verzekeringen

Maak de persona menselijk, herkenbaar en gedetailleerd. Gebruik Nederlandse cultuur en context.

Formatteer je antwoord als JSON met de volgende structuur:
{{
    "name": "...",
    "age": ...,
    "job": "...",
    "bio": "...",
    "insurance_attitude": "...",
    "quote": "...",
    "values": ["...", "...", "..."],
    "concerns": ["...", "...", "..."],
    "emoji": "..." (één emoji die deze persoon representeert)
}}
"""
    return prompt


def generate_personas(cluster_stats_df: pd.DataFrame, client: OpenAI) -> List[Dict]:
    """
    Generate personas for each cluster using the LLM.

    Args:
        cluster_stats_df: DataFrame with cluster statistics
        client: OpenAI client instance

    Returns:
        List of persona dictionaries
    """
    personas = []

    for idx, row in cluster_stats_df.iterrows():
        cluster_dict = row.to_dict()
        prompt = generate_persona_prompt(cluster_dict)

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Je bent een expert in customer persona ontwikkeling. Antwoord altijd in geldig JSON formaat."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1000
            )

            persona_json = response.choices[0].message.content
            # Extract JSON from markdown code blocks if present
            if "```json" in persona_json:
                persona_json = persona_json.split("```json")[1].split("```")[0].strip()
            elif "```" in persona_json:
                persona_json = persona_json.split("```")[1].split("```")[0].strip()

            persona = json.loads(persona_json)
            persona['cluster_id'] = idx
            persona['cluster_stats'] = cluster_dict
            personas.append(persona)

        except Exception as e:
            st.error(f"Error generating persona for {cluster_dict['Cluster']}: {str(e)}")
            continue

    return personas


def test_proposition_with_persona(proposition: str, persona: Dict, client: OpenAI) -> Dict:
    """
    Test a market research proposition with a specific persona.

    Args:
        proposition: The proposition/hypothesis to test
        persona: Persona dictionary
        client: OpenAI client instance

    Returns:
        Dictionary with rating and reasoning
    """
    prompt = f"""Je bent {persona['name']}, {persona['age']} jaar oud, werkzaam als {persona['job']}.

JOUW ACHTERGROND:
{persona['bio']}

JOUW HOUDING TEN OPZICHTE VAN VERZEKERINGEN:
{persona['insurance_attitude']}

JOUW WAARDEN: {', '.join(persona['values'])}
JOUW ZORGEN: {', '.join(persona['concerns'])}

Een verzekeringsmaatschappij presenteert het volgende voorstel:
"{proposition}"

Reageer als {persona['name']} op dit voorstel:
1. Geef een rating van 1-10 (1 = zeer negatief, 10 = zeer positief)
2. Leg uit waarom je deze rating geeft vanuit jouw perspectief
3. Noem specifieke voor- en nadelen die jij ziet
4. Geef aan of je dit zou accepteren of niet

Formatteer je antwoord als JSON:
{{
    "rating": <getal 1-10>,
    "reaction": "...",
    "pros": ["...", "..."],
    "cons": ["...", "..."],
    "decision": "accept" of "reject",
    "reasoning": "..."
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Je bent {persona['name']}. Blijf volledig in karakter en reageer authentiek vanuit deze persona."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )

        result_json = response.choices[0].message.content
        if "```json" in result_json:
            result_json = result_json.split("```json")[1].split("```")[0].strip()
        elif "```" in result_json:
            result_json = result_json.split("```")[1].split("```")[0].strip()

        result = json.loads(result_json)
        result['persona_name'] = persona['name']
        return result

    except Exception as e:
        return {
            "rating": 5,
            "reaction": f"Error: {str(e)}",
            "pros": [],
            "cons": [],
            "decision": "unknown",
            "reasoning": "Could not process response",
            "persona_name": persona['name']
        }


def chat_with_persona(message: str, persona: Dict, chat_history: List, client: OpenAI) -> str:
    """
    Have a conversation with a specific persona.

    Args:
        message: User's message
        persona: Persona dictionary
        chat_history: List of previous messages
        client: OpenAI client instance

    Returns:
        Persona's response
    """
    system_prompt = f"""Je bent {persona['name']}, een {persona['age']}-jarige {persona['job']}.

JOUW PERSOONLIJKHEID EN ACHTERGROND:
{persona['bio']}

JOUW HOUDING TEN OPZICHTE VAN VERZEKERINGEN:
{persona['insurance_attitude']}

JOUW KERNWAARDEN: {', '.join(persona['values'])}
JOUW ZORGEN: {', '.join(persona['concerns'])}

JOUW KENMERKENDE UITSPRAAK: "{persona['quote']}"

BELANGRIJKE INSTRUCTIES:
- Blijf volledig in karakter als {persona['name']}
- Gebruik een natuurlijke, authentieke Nederlandse spreektaal
- Laat jouw persoonlijkheid, waarden en zorgen doorschemeren in je antwoorden
- Reageer zoals deze persona zou reageren, niet als een generieke assistant
- Wees eerlijk over je mening, ook als die kritisch is
- Gebruik zo nu en dan verwijzingen naar je dagelijks leven of achtergrond
"""

    messages = [{"role": "system", "content": system_prompt}]

    # Add chat history
    for hist in chat_history[-6:]:  # Keep last 6 messages for context
        messages.append(hist)

    # Add current message
    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.8,
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Sorry, ik kan nu even niet reageren. Probeer het later nog eens. (Error: {str(e)})"


# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("⚙️ Configuratie")

    st.markdown("---")

    # API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Voer je OpenAI API key in om LLM functionaliteit te gebruiken"
    )

    if api_key:
        st.success("✅ API Key ingesteld")
        client = initialize_openai_client(api_key)
    else:
        st.warning("⚠️ Voer een API key in om te starten")
        client = None

    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Customer Data (CSV)",
        type=['csv'],
        help="Upload een CSV bestand met klantdata"
    )

    if uploaded_file:
        st.success(f"✅ Bestand geladen: {uploaded_file.name}")

        # CSV format configuration
        st.markdown("#### CSV Formaat Instellingen")

        separator_option = st.radio(
            "Veld Separator",
            options=["Comma (,)", "Semicolon (;)"],
            index=1,  # Default to semicolon for Dutch files
            help="Kies het scheidingsteken dat in je CSV wordt gebruikt"
        )

        decimal_option = st.radio(
            "Decimaal Teken",
            options=["Dot (.)", "Comma (,)"],
            index=1,  # Default to comma for Dutch files
            help="Kies het decimale scheidingsteken voor getallen"
        )

        # Map options to actual separators
        separator = "," if separator_option == "Comma (,)" else ";"
        decimal = "." if decimal_option == "Dot (.)" else ","

    st.markdown("---")

    # Number of clusters
    n_clusters = st.slider(
        "Aantal Clusters",
        min_value=3,
        max_value=8,
        value=5,
        help="Kies het aantal klantsegmenten om te identificeren"
    )

    st.markdown("---")
    st.markdown("### 📊 Over deze Tool")
    st.markdown("""
    Deze tool gebruikt:
    - **K-Means Clustering** voor segmentatie
    - **LLM's** voor persona generatie
    - **AI-gedreven** marktonderzoek
    """)


# ==================== MAIN CONTENT ====================
st.markdown('<h1 class="main-header">🎯 Synthetic Customer Persona & Market Research Tool</h1>', unsafe_allow_html=True)

if not uploaded_file:
    st.info("👈 Upload een CSV bestand via de sidebar om te beginnen")
    st.stop()

if not client:
    st.warning("👈 Voer een OpenAI API key in via de sidebar om door te gaan")
    st.stop()

# Load data with dynamic separator and decimal settings
try:
    df = pd.read_csv(uploaded_file, sep=separator, decimal=decimal)
    st.success(f"✅ Data geladen: {len(df)} rijen, {len(df.columns)} kolommen")
except Exception as e:
    st.error(f"❌ Error bij het laden van data: {str(e)}")
    st.error("💡 Tip: Controleer of de separator en decimaal instellingen correct zijn in de sidebar.")
    st.stop()

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Data & Segmentatie", "👥 Persona Generatie", "🔬 Market Research Simulator"])

# ==================== TAB 1: DATA & SEGMENTATION ====================
with tab1:
    st.header("Data Ingestion & Segmentatie")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

    with col2:
        st.subheader("Dataset Info")
        st.metric("Total Rijen", f"{len(df):,}")
        st.metric("Total Kolommen", len(df.columns))
        st.metric("Missing Values", df.isnull().sum().sum())

    st.markdown("---")

    if st.button("🚀 Start Clustering Analysis", type="primary", use_container_width=True):
        with st.spinner("Preprocessing data..."):
            X_scaled, df_clean, preprocessor = preprocess_data(df)
            st.session_state.processed_data = df_clean

        with st.spinner(f"Performing K-Means clustering (k={n_clusters})..."):
            kmeans = perform_clustering(X_scaled, n_clusters)
            st.session_state.cluster_model = kmeans

        with st.spinner("Calculating cluster statistics..."):
            cluster_stats = calculate_cluster_statistics(df_clean, kmeans.labels_)
            st.session_state.cluster_stats = cluster_stats

        st.success("✅ Clustering compleet!")

    # Display cluster statistics
    if st.session_state.cluster_stats is not None:
        st.subheader("📈 Cluster Overzicht")

        cluster_stats_df = st.session_state.cluster_stats

        # Display size distribution
        fig = px.pie(
            cluster_stats_df,
            names='Cluster',
            values='Size',
            title='Verdeling van Klanten over Clusters',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display detailed statistics
        st.subheader("Cluster Karakteristieken")

        # Format the dataframe for better display
        display_df = cluster_stats_df.copy()
        for col in display_df.columns:
            if col.startswith('Avg_') and display_df[col].dtype in [np.float64, np.int64]:
                display_df[col] = display_df[col].round(2)

        st.dataframe(display_df, use_container_width=True, height=400)


# ==================== TAB 2: PERSONA GENERATION ====================
with tab2:
    st.header("Persona Generatie")

    if st.session_state.cluster_stats is None:
        st.warning("⚠️ Voer eerst de clustering analyse uit in Tab 1")
        st.stop()

    st.markdown("""
    Genereer rijke, gedetailleerde buyer personas op basis van de geïdentificeerde klantsegmenten.
    Elke persona krijgt een naam, achtergrond, en unieke persoonlijkheid.
    """)

    st.markdown("---")

    if st.button("✨ Genereer Personas", type="primary", use_container_width=True):
        with st.spinner("Personas worden gegenereerd... Dit kan even duren."):
            personas = generate_personas(st.session_state.cluster_stats, client)
            st.session_state.personas = personas

        if personas:
            st.success(f"✅ {len(personas)} personas succesvol gegenereerd!")
        else:
            st.error("❌ Fout bij het genereren van personas")

    # Display personas
    if st.session_state.personas:
        st.markdown("---")
        st.subheader("👥 Gegenereerde Personas")

        for persona in st.session_state.personas:
            with st.container():
                col1, col2 = st.columns([1, 4])

                with col1:
                    st.markdown(f"<div style='font-size: 80px; text-align: center;'>{persona.get('emoji', '👤')}</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown(f"### {persona['name']}")
                    st.markdown(f"**{persona['age']} jaar | {persona['job']}**")
                    st.markdown(f"*\"{persona['quote']}\"*")

                # Expandable details
                with st.expander("📋 Volledige Persona Details"):
                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.markdown("**Bio**")
                        st.write(persona['bio'])

                        st.markdown("**Waarden**")
                        for value in persona['values']:
                            st.markdown(f"- {value}")

                    with col_b:
                        st.markdown("**Houding t.o.v. Verzekeringen**")
                        st.write(persona['insurance_attitude'])

                        st.markdown("**Zorgen & Angsten**")
                        for concern in persona['concerns']:
                            st.markdown(f"- {concern}")

                st.markdown("---")


# ==================== TAB 3: MARKET RESEARCH SIMULATOR ====================
with tab3:
    st.header("🔬 Market Research Simulator")

    if not st.session_state.personas:
        st.warning("⚠️ Genereer eerst personas in Tab 2")
        st.stop()

    # Mode selection
    mode = st.radio(
        "Kies een modus:",
        ["📊 Panel Test (Macro View)", "💬 Deep Dive Interview (Micro View)"],
        horizontal=True
    )

    st.markdown("---")

    # ========== MODE A: PANEL TEST ==========
    if mode == "📊 Panel Test (Macro View)":
        st.subheader("Panel Test: Test je Propositie")

        st.markdown("""
        Voer een hypothese of propositie in en zie hoe alle personas erop reageren.
        Ideaal voor het snel testen van concepten.
        """)

        proposition = st.text_area(
            "Propositie / Hypothese",
            placeholder="Bijvoorbeeld: We verhogen de premies met 5% maar bieden gratis pechhulp aan.",
            height=100
        )

        if st.button("🚀 Test met Alle Personas", type="primary", disabled=not proposition):
            results = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, persona in enumerate(st.session_state.personas):
                status_text.text(f"Testing met {persona['name']}...")
                result = test_proposition_with_persona(proposition, persona, client)
                results.append(result)
                progress_bar.progress((idx + 1) / len(st.session_state.personas))

            status_text.empty()
            progress_bar.empty()

            # Display results
            st.markdown("---")
            st.subheader("📊 Resultaten")

            # Create rating chart
            fig = go.Figure(data=[
                go.Bar(
                    x=[r['persona_name'] for r in results],
                    y=[r['rating'] for r in results],
                    marker_color=[
                        '#d32f2f' if r['rating'] < 5 else '#ff9800' if r['rating'] < 7 else '#4caf50'
                        for r in results
                    ],
                    text=[r['rating'] for r in results],
                    textposition='auto',
                )
            ])

            fig.update_layout(
                title="Ratings per Persona (1-10)",
                xaxis_title="Persona",
                yaxis_title="Rating",
                yaxis_range=[0, 10],
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_rating = np.mean([r['rating'] for r in results])
                st.metric("Gemiddelde Rating", f"{avg_rating:.1f}/10")
            with col2:
                accept_pct = sum([1 for r in results if r['decision'] == 'accept']) / len(results) * 100
                st.metric("Acceptatie Rate", f"{accept_pct:.0f}%")
            with col3:
                reject_pct = sum([1 for r in results if r['decision'] == 'reject']) / len(results) * 100
                st.metric("Afwijzing Rate", f"{reject_pct:.0f}%")

            # Detailed feedback
            st.markdown("---")
            st.subheader("💬 Gedetailleerde Feedback")

            for result in results:
                with st.expander(f"{result['persona_name']} - Rating: {result['rating']}/10 - {result['decision'].upper()}"):
                    st.markdown(f"**Reactie:** {result['reaction']}")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Voordelen:**")
                        for pro in result.get('pros', []):
                            st.markdown(f"✅ {pro}")

                    with col_b:
                        st.markdown("**Nadelen:**")
                        for con in result.get('cons', []):
                            st.markdown(f"❌ {con}")

                    st.markdown(f"**Redenering:** {result.get('reasoning', 'N/A')}")

    # ========== MODE B: DEEP DIVE INTERVIEW ==========
    else:
        st.subheader("Deep Dive Interview: 1-op-1 Gesprek")

        st.markdown("""
        Selecteer een persona en voer een diepgaand gesprek. De persona blijft volledig in karakter
        en reageert authentiek vanuit hun perspectief.
        """)

        # Persona selection
        persona_names = [p['name'] for p in st.session_state.personas]
        selected_persona_name = st.selectbox("Kies een Persona", persona_names)

        selected_persona = next(p for p in st.session_state.personas if p['name'] == selected_persona_name)

        # Display persona info
        with st.container():
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"<div style='font-size: 60px;'>{selected_persona.get('emoji', '👤')}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"### {selected_persona['name']}")
                st.markdown(f"*{selected_persona['age']} jaar | {selected_persona['job']}*")
                st.markdown(f"_{selected_persona['quote']}_")

        st.markdown("---")

        # Initialize chat history for this persona
        if selected_persona_name not in st.session_state.chat_history:
            st.session_state.chat_history[selected_persona_name] = []

        # Chat interface
        chat_container = st.container()

        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history[selected_persona_name]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        # Chat input
        user_message = st.chat_input("Type je vraag of opmerking...")

        if user_message:
            # Add user message to history
            st.session_state.chat_history[selected_persona_name].append({
                "role": "user",
                "content": user_message
            })

            # Get persona response
            with st.spinner(f"{selected_persona['name']} denkt na..."):
                response = chat_with_persona(
                    user_message,
                    selected_persona,
                    st.session_state.chat_history[selected_persona_name],
                    client
                )

            # Add assistant message to history
            st.session_state.chat_history[selected_persona_name].append({
                "role": "assistant",
                "content": response
            })

            # Rerun to display new messages
            st.rerun()

        # Clear chat button
        if st.button("🗑️ Wis Gesprek"):
            st.session_state.chat_history[selected_persona_name] = []
            st.rerun()


# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🎯 Synthetic Customer Persona & Market Research Tool</p>
    <p>Powered by K-Means Clustering & Advanced LLM Technology</p>
</div>
""", unsafe_allow_html=True)
