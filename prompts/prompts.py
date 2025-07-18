from langchain.core import  SystemMessage

UserSummary = SystemMessage(
    content="""
    You are a assistant that summarizes user information based on the provided user data.""",
)