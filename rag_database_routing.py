def init_session_state():
    """Initialize session state variables"""
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
    if 'qdrant_url' not in st.session_state:
        st.session_state.qdrant_url = ""
    if 'qdrant_api_key' not in st.session_state:
        st.session_state.qdrant_api_key = ""
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'databases' not in st.session_state:
        st.session_state.databases = {}

init_session_state()

DatabaseType = Literal["products", "support", "finance"]
PERSIST_DIRECTORY = "db_storage"

@dataclass
class CollectionConfig: 
    name: str
    description: str
    collection_name: str  # This will be used as Qdrant collection name
    
    # Collection configurations
    COLLECTIONS: Dict[DatabaseType, CollectionConfig] = {
    "products": CollectionConfig(
        name="Product Information",
        description="Product details, specifications, and features",
        collection_name="products_collection"
    ),
    "support": CollectionConfig(
        name="Customer Support & FAQ",
        description="Customer support information, frequently asked questions, and guides",
        collection_name="support_collection"
    ),
    "finance": CollectionConfig(
        name="Financial Information",
        description="Financial data, revenue, costs, and liabilities",
        collection_name="finance_collection"
    )
}

def initialize_models():
    """Initialize OpenAI models and Qdrant client"""
    if (st.session_state.openai_api_key and 
        st.session_state.qdrant_url and 
        st.session_state.qdrant_api_key):
        
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
        st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        st.session_state.llm = ChatOpenAI(temperature=0)