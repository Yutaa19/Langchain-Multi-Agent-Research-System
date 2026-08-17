from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.messages import AIMessage
from src.tools.tools import search_web, scrape_web
from dotenv import load_dotenv
import re
import os

load_dotenv()


class QwenOutputParser(BaseOutputParser[str]):
    """
    Output parser khusus untuk model Qwen3 di Groq.

    Masalah: Qwen3 memisahkan output menjadi dua bagian:
      - `reasoning_content` : chain-of-thought / thinking panjang
      - `content`           : jawaban final yang bersih

    StrOutputParser() menggunakan TextAccessor yang menggabungkan keduanya,
    sehingga saat di-strip `<think>` bisa memotong konten di tempat yang salah.
    Parser ini langsung baca AIMessage.content untuk hasil yang bersih.
    """

    @property
    def _type(self) -> str:
        return "qwen_output_parser"

    def parse(self, text: str) -> str:
        # Strip <think>...</think> jika ada sisa di content field
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return cleaned

    def invoke(self, input, config=None, **kwargs):
        if isinstance(input, AIMessage):
            # Baca content langsung (bukan TextAccessor)
            content = input.content if isinstance(input.content, str) else str(input.content)
            # Fallback ke reasoning_content jika content kosong
            if not content.strip():
                content = input.additional_kwargs.get("reasoning_content", "")
            return self.parse(content)
        return self.parse(str(input))

class GroqLLM(ChatGroq):
    """
    ChatGroq wrapper yang meng-inject parallel_tool_calls=False secara otomatis.
    """
    def bind_tools(self, tools, **kwargs):
        kwargs.setdefault("parallel_tool_calls", False)
        return super().bind_tools(tools, **kwargs)

llm = GroqLLM(
    model="openai/gpt-oss-safeguard-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

_SEARCH_AGENT_PROMPT = (
    "You are a research assistant with ONE tool available: 'search_web'. "
    "Use 'search_web' to search for information based on a query. "
    "NEVER call any tool other than 'search_web'. "
    "Do NOT call 'scrape_web', 'open_url', 'browse', or any other tool name."
)

_READING_AGENT_PROMPT = (
    "You are a research assistant with TWO tools available: 'search_web' and 'scrape_web'. "
    "Use 'search_web' to find URLs, then use 'scrape_web' to read the content of a URL. "
    "NEVER call tools named 'open_url', 'browse', 'visit_url', or any other name. "
    "Only call 'search_web' or 'scrape_web'."
)

def create_search_agent():
    return create_agent(
        model=llm,
        tools=[search_web],
        system_prompt=_SEARCH_AGENT_PROMPT,
    )

def create_reading_agent():
    return create_agent(
        model=llm,
        tools=[scrape_web, search_web],
        system_prompt=_READING_AGENT_PROMPT,
    )

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Anda adalah penulis riset. Buat laporan yang jelas dan terstruktur."),
    ("human", """Buat laporan penelitian tentang topik berikut.

Topik: {topic}

Hasil Penelitian:
{research}

Struktur laporan:
- Pendahuluan
- Temuan Utama (minimal 3 poin)
- Kesimpulan
- Sumber (URL)"""),
])


writer_chain = writer_prompt | llm | QwenOutputParser()


critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "Anda adalah kritikus penelitian yang konstruktif dan jujur."),
    ("human", """Tinjau laporan berikut secara kritis.

Laporan:
{report}

Format respons:
Skor: X/10

Kelebihan:
- ...

Aspek yang perlu ditingkatkan:
- ...

Kesimpulan:
..."""),
])

critic_chain = critic_prompt | llm | QwenOutputParser()