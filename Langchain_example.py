# pip install langchain langchain-openai
import os
os.environ["OPENAI_API_KEY"] = "your-key"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("用一句话解释什么是{concept}")
model = ChatOpenAI(model="gpt-4o-mini")

# LCEL：用 | 把步骤串成链
chain = prompt | model | StrOutputParser()

print(chain.invoke({"concept": "LangGraph"}))