# pip install langgraph langchain langchain-openai
import os
os.environ["OPENAI_API_KEY"] = "your-key"

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

# ① 定义状态：整个图共享的数据
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ② 定义一个工具
@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    return f"{city}今天晴，25°C"

# ③ 定义节点
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([get_weather])

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# ④ 条件路由：模型想调工具就去 tools 节点，否则结束
def should_continue(state: State):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END

# ⑤ 组装图
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode([get_weather]))
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", should_continue)
builder.add_edge("tools", "chatbot")   # 工具执行完回到 chatbot，形成循环

graph = builder.compile()

# ⑥ 运行
for chunk in graph.stream({"messages": [("user", "北京今天天气怎么样？")]}):
    print(chunk)