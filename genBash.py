from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
import os
import subprocess
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['GROQ_API_KEY']="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"
os.environ['TOKENIZERS_PARALLELISM']='true'
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db=Chroma(persist_directory='/home/vineeth/Desktop/code/projects/genBash/chroma',embedding_function=embeddings)

model=ChatGroq(model='llama3-8b-8192')

#getting some background information of the computer: username, current working directory, hostname, current directory information
userName=subprocess.run(['whoami'], capture_output=True, text=True).stdout
hostName=subprocess.run(['hostname'], capture_output=True, text=True).stdout
global currentDir
currentDir = subprocess.run(['pwd'], capture_output=True, text=True).stdout
currentDirInfo=subprocess.run(['ls', '-l'], capture_output=True, text=True).stdout
userInfo=f"Username: {userName} \nHostname: {hostName} \nCurrent Directory: {currentDir} \nCurrent Directory Information: {currentDirInfo}"

# template=ChatPromptTemplate.from_messages()

@tool
def executeCommand(text:str)->str:
    """Execute terminal commamd
    
    Args:
        text: command to be executed
    """
    textC=text.split()
    global currentDir
    print("Command:",text)
    result = subprocess.run('cd '+currentDir[:-1]+';'+text, capture_output=True, text=True,shell=True)
    print(result.stdout)
    for i in range(0,len(textC)):
        if(textC[i]=='cd'):
            if(i+1<len(textC)):
                currentDir=subprocess.run(f'cd {textC[i+1]}; pwd', capture_output=True, text=True, shell=True).stdout
            else:
                currentDir=subprocess.run('cd;pwd',capture_output=True, text=True,shell=True).stdout
            print("current directory opened:"+currentDir)
tools=[executeCommand]
model_with_tools=model.bind_tools(tools=tools)
executeCommand('clear')
while (1):
    text=input(userName[:-1]+'@'+hostName[:-1]+currentDir[:-1]+"$ ")
    try:    
        results = db.similarity_search_with_relevance_scores(text, k=3)
        template=f"You are a Linux command-line assistant. Based on the provided documentation and your knowledge, generate the most appropriate Linux command for the user's request and execute it with the executeCommand function.\nBackground Information about the user: {userInfo}\nUser Input: {text}\nRelevant Information: Source1:{results[0][0].metadata}\nSource2:{results[1][0].metadata}\nSource3:{results[2][0].metadata}"
        result=model_with_tools.invoke(template)
        if result.tool_calls:
            executeCommand.invoke(result.tool_calls[0]['args'])
    except:
         print("something went wrong. Please try again")