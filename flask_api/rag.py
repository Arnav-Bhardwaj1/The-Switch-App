import os
from typing import List, Optional, Tuple

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic.v1 import BaseModel, Field


def get_simulated_data() -> Tuple[list, str, str]:
    user_history = [
        {"timestamp": "2025-09-01T10:00:00Z", "mood": 3, "energy": 2, "journal_entry": "Feeling overwhelmed with the new semester. So much to do."},
        {"timestamp": "2025-09-02T11:00:00Z", "mood": 4, "energy": 5, "journal_entry": "Did a 5-minute breathing exercise this morning. It actually helped a bit. Felt calmer."},
        {"timestamp": "2025-09-03T09:30:00Z", "mood": 2, "energy": 2, "journal_entry": "Exam pressure is mounting. I skipped my morning walk."},
        {"timestamp": "2025-09-04T12:00:00Z", "mood": 5, "energy": 6, "journal_entry": "Talked to a friend and went for a walk. Feeling much better and more focused now."},
        {"timestamp": "2025-09-08T14:00:00Z", "mood": 2, "energy": 3, "journal_entry": "Another stressful week starting. It feels like the same pressure as last week."},
    ]
    campus_resources_text = (
        """
        Document Title: University Counseling Center FAQ
        Content: The University Counseling Center (UCC) offers free, confidential sessions for all enrolled students...
        """
    )
    academic_articles_text = (
        """
        Document Title: The Benefit of Micro-Habits
        Content: Micro-habits are small, repeatable actions that require minimal motivation...
        """
    )
    return user_history, campus_resources_text, academic_articles_text


class GroundedAnswer(BaseModel):
    answer: str = Field(description="The empathetic, helpful answer to the user's query.")
    actions: Optional[List[str]] = Field(description="A list of 1-3 suggested micro-habits or next steps.")
    sources: Optional[List[str]] = Field(description="A list of source documents used to generate the answer, for citation.")
    confidence: float = Field(description="A confidence score (0.0 to 1.0) on how well the context answered the query.")


class TaruRAG:
    def __init__(self, documents: List[Document], api_key: str):
        self.documents = documents
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.3,
            google_api_key=api_key,
            convert_system_message_to_human=True,
        )
        self.llm_with_structure = self.llm.with_structured_output(GroundedAnswer)
        self.vectorstore = self._create_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self.chain = self._create_rag_chain()

    @classmethod
    def from_raw_texts(cls, user_history: list, campus_resources: str, academic_articles: str, api_key: str) -> "TaruRAG":
        docs: List[Document] = []
        for entry in user_history:
            doc = Document(
                page_content=entry["journal_entry"],
                metadata={
                    "source": "user_journal",
                    "timestamp": entry["timestamp"],
                    "mood": entry["mood"],
                    "energy": entry["energy"],
                },
            )
            docs.append(doc)

        # Prepare and split long texts
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        with open("campus_resources.txt", "w", encoding="utf-8") as f:
            f.write(campus_resources)
        with open("academic_articles.txt", "w", encoding="utf-8") as f:
            f.write(academic_articles)
        campus_docs = TextLoader("campus_resources.txt").load_and_split(text_splitter)
        academic_docs = TextLoader("academic_articles.txt").load_and_split(text_splitter)
        try:
            os.remove("campus_resources.txt")
            os.remove("academic_articles.txt")
        except OSError:
            pass

        all_documents = docs + campus_docs + academic_docs
        return cls(all_documents, api_key)

    def _create_vectorstore(self) -> FAISS:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(self.documents, embeddings)

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(
            [f"--- Document ---\nMetadata: {doc.metadata}\nContent: {doc.page_content}" for doc in docs]
        )

    def _create_rag_chain(self):
        template = (
            """
            You are Taru, an empathetic and supportive wellness coach for students.
            Use the provided context to answer the user's question with compassion and clarity.
            Provide at most three actionable, specific next steps. If the context is weak,
            state your uncertainty and suggest general wellness resources.

            Context: {context}
            Question: {question}
            """
        )
        prompt = ChatPromptTemplate.from_template(template)
        return (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}  # type: ignore
            | prompt
            | self.llm_with_structure
        )

    def query(self, question: str) -> GroundedAnswer:
        return self.chain.invoke(question)


