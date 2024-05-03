import os
import sys
import taipy as tp
from taipy.gui import notify, State, Markdown, navigate, Icon, Gui
from datetime import date
import yfinance as yf
from prophet import Prophet
import pandas as pd
from plotly import graph_objects as go

from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.core.agent import AgentRunner
import requests
import finnhub


### BEGIN OPEN AI ######################################################################################
import openai

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
client = None
context = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, Hello!\nAI: I am an AI created by OpenAI. How can I help you today? "
conversation = {
    "Conversation": ["Hello!", "Hi! I am CG-Agent. How can I help you today?"]
}
current_user_message = ""
past_conversations = []
selected_conv = None
selected_row = [1]
past_prompts = []
selected_stock = 'AAPL'
agent_response = ""

def on_init(state: State) -> None:
    """
    Initialize the app.

    Args:
        - state: The current state of the app.
    """
    state.context = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, Hello!\nAI: I am an AI created by OpenAI. How can I help you today? "
    state.conversation = {
        "Conversation": ["Hello!", "Hi! I am CG-Agent. How can I help you today?"]
    }
    state.current_user_message = ""
    state.past_conversations = []
    state.selected_conv = None
    state.selected_row = [1]


def request(state: State, prompt: str) -> str:
    """
    Send a prompt to the GPT-4 API and return the response.

    Args:
        - state: The current state of the app.
        - prompt: The prompt to send to the API.

    Returns:
        The response from the API.
    """
    response = state.client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{prompt}",
            }
        ],
        model="gpt-4-turbo-preview",
    )
    
    
    # Call Finnhub agent
    #print(f"User message : {state.current_user_message}")
    ticker_name = get_ticker_name(state)
    response = call_finnhub_agent(state.current_user_message)
    response = response.replace("assistant: ", "")
    
    #return response.choices[0].message.content
    return response


def update_context(state: State) -> None:
    """
    Update the context with the user's message and the AI's response.

    Args:
        - state: The current state of the app.
    """
    state.context += f"Human: \n {state.current_user_message}\n\n AI:"
    #answer = request(state, state.context).replace("\n", "")
    answer = request(state, state.context)
    state.context += answer
    state.selected_row = [len(state.conversation["Conversation"]) + 1]
    return answer


def send_message(state: State) -> None:
    """
    Send the user's message to the API and update the context.

    Args:
        - state: The current state of the app.
    """
    notify(state, "info", "Sending message...")
    
    answer = update_context(state)
    print(f"\nAgent response received: {answer}\n")
    conv = state.conversation._dict.copy()
    conv["Conversation"] += [state.current_user_message, answer]
    state.current_user_message = ""
    state.conversation = conv
    state.refresh("conversation")  # Refresh UI elements bound to state variables
    notify(state, "success", "Response received!")


def get_ticker_name(state: State) -> None:
    """
    Send the user's message to the API and get stock ticker name.

    Args:
        - state: The current state of the app.
    """
    prompt_msg = "message : " + str(state.current_user_message) + "query: Check if there are any company names mentioned in the message and determine if their respective stock ticker symbols are listed on the New York Stock Exchange (NYSE). If a company name with a corresponding NYSE ticker symbol is found, respond with the ticker symbol as a single word. Otherwise, respond with 'NONE'"
    #print("\nGetting ticker name:")
    agent_response = agent.chat(prompt_msg)
    #print(f"Ticker = {agent_response}\n\n")
    agent_response = str(agent_response).replace("\n", " ").replace('\"', "\'").replace(".", "").split(" ")[-1]
    print(f"Ticker symbol found = {agent_response}\n\n")
    
    if agent_response != "NONE":
        state.selected_stock = agent_response



def style_conv(state: State, idx: int, row: int) -> str:
    """
    Apply a style to the conversation table depending on the message's author.

    Args:
        - state: The current state of the app.
        - idx: The index of the message in the table.
        - row: The row of the message in the table.

    Returns:
        The style to apply to the message.
    """
    if idx is None:
        return None
    elif idx % 2 == 0:
        return "user_message"
    else:
        return "gpt_message"


def on_exception(state, function_name: str, ex: Exception) -> None:
    """
    Catches exceptions and notifies user in Taipy GUI

    Args:
        state (State): Taipy GUI state
        function_name (str): Name of function where exception occured
        ex (Exception): Exception
    """
    notify(state, "error", f"An error occured in {function_name}: {ex}")


def reset_chat(state: State) -> None:
    """
    Reset the chat by clearing the conversation.

    Args:
        - state: The current state of the app.
    """
    state.past_conversations = state.past_conversations + [
        [len(state.past_conversations), state.conversation]
    ]
    state.conversation = {
        "Conversation": ["Hello!", "Hi! I am CG-Agent. How can I help you today?"]
    }


def tree_adapter(item: list) -> [str, str]:
    """
    Converts element of past_conversations to id and displayed string

    Args:
        item: element of past_conversations

    Returns:
        id and displayed string
    """
    identifier = item[0]
    if len(item[1]["Conversation"]) > 3:
        return (identifier, item[1]["Conversation"][2][:50] + "...")
    return (item[0], "Empty conversation")


def select_conv(state: State, var_name: str, value) -> None:
    """
    Selects conversation from past_conversations

    Args:
        state: The current state of the app.
        var_name: "selected_conv"
        value: [[id, conversation]]
    """
    state.conversation = state.past_conversations[value[0][0]][1]
    state.context = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, Hello!\nAI: I am an AI created by OpenAI. How can I help you today? "
    for i in range(2, len(state.conversation["Conversation"]), 2):
        state.context += f"Human: \n {state.conversation['Conversation'][i]}\n\n AI:"
        state.context += state.conversation["Conversation"][i + 1]
    state.selected_row = [len(state.conversation["Conversation"]) + 1]

### END OPEN AI ########################################################################################


### BEGIN FINNHUB ########################################################################################

finnhub_client = finnhub.Client(api_key="colv4jpr01qqra7ga6q0colv4jpr01qqra7ga6qg")

def fin_companynews(ticker: str, from_date: str = "YYYY-MM-DD", to_date: str = "YYYY-MM-DD") -> pd.DataFrame:
  #this function returns the symbol in case the user types name of stock but doesn't know the ticker
  return pd.DataFrame(finnhub_client.company_news({ticker},{from_date},{to_date}))

def fin_basicfinancials(b:str):#should be a tiker
  #this function returns the basic financials of a stock. It is necessary to ensure that ticker is passed to this from the fin_getticker function if the user doesn't remember the ticker
  return pd.DataFrame(finnhub_client.company_basic_financials({b},'all'))


# Tool wrapper

def init_finnhub():
    global agent
    print("FinnHub Agent initialized!")
    fin_companynews_tool = FunctionTool.from_defaults(fn=fin_companynews)
    fin_basicfinancials_tool = FunctionTool.from_defaults(fn=fin_basicfinancials)

    llm = OpenAI(model="gpt-4-turbo")

    agent_worker = FunctionCallingAgentWorker.from_tools(
        [fin_basicfinancials_tool,fin_companynews_tool],
        llm=llm,
        verbose=False,
        allow_parallel_tool_calls=True
    )
    agent = AgentRunner(agent_worker)

def call_finnhub_agent(usr_msg: str) -> None:
    global agent_response
    prompt_msg = usr_msg + " Give reasons by using all tools available to you"
    #print("fInvoking FinnHub agent with prompt {prompt_msg}")
    agent_response = agent.chat(prompt_msg)

    return str(agent_response)

### END FINNHUB ########################################################################################


# Parameters for retrieving the stock data
start_date = "2015-01-01"
end_date = date.today().strftime("%Y-%m-%d")
selected_stock = 'AAPL'
n_years = 1


def get_stock_data(ticker, start, end):
    ticker_data = yf.download(ticker, start, end)  # downloading the stock data from START to TODAY
    ticker_data.reset_index(inplace=True)  # put date in the first column
    ticker_data['Date'] = pd.to_datetime(ticker_data['Date']).dt.tz_localize(None)
    return ticker_data

def get_data_from_range(state):
    print(f"GENERATING HIST DATA FOR {state.selected_stock}")
    start_date = state.start_date if type(state.start_date)==str else state.start_date.strftime("%Y-%m-%d")
    end_date = state.end_date if type(state.end_date)==str else state.end_date.strftime("%Y-%m-%d")

    state.data = get_stock_data(state.selected_stock, start_date, end_date)
    if len(state.data) == 0:
        notify(state, "error", f"Not able to download data {state.selected_stock} from {start_date} to {end_date}")
        return
    notify(state, 's', 'Historical data has been updated!')
    notify(state, 'w', 'Deleting previous predictions...')
    state.forecast = pd.DataFrame(columns=['Date', 'Lower', 'Upper'])
    forecast_display(state)


def create_candlestick_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data['Date'],
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlestick'))
    fig.update_layout(margin=dict(l=30, r=30, b=30, t=30), xaxis_rangeslider_visible=False)
    return fig


def generate_forecast_data(data, n_years):
    # FORECASTING
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})  # This is the format that Prophet accepts

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=n_years * 365)
    fc = m.predict(future)[['ds', 'yhat_lower', 'yhat_upper']].rename(columns={"ds": "Date", "yhat_lower": "Lower", "yhat_upper": "Upper"})
    print("Process Completed!")
    return fc


def forecast_display(state):
    notify(state, 'i', 'Predicting...')
    state.forecast = generate_forecast_data(state.data, state.n_years)
    notify(state, 's', 'Prediction done! Forecast data has been updated!')



#### Getting the data, make initial forcast and build a front end web-app with Taipy GUI
data = get_stock_data(selected_stock, start_date, end_date)
forecast = generate_forecast_data(data, n_years)

show_dialog = False
image_path = '.\images\IISc_TeamMembers.png'
arrow_img = '.\images\ArrowHead.svg'

partial_md = "<|{forecast}|table|>"
dialog_md = "<|{show_dialog}|dialog|partial={partial}|title=Forecast Data|on_action={lambda state: state.assign('show_dialog', False)}|>"

agent_response = ""

root_page_md = dialog_md + """
<|toggle|theme|>

<|container|
# CAPITAL G***AI***{: .color-primary}N
-----------------------------------------------------------------
## Hello, Welcome to your personalized ***Financial Advisor***{: .color-primary}, powered by AI.
|>


<|container|
<|layout|columns=1|gap=10px|class_name=card mt1|

<|
<|part|class_name=p2 align-item-bottom table|
<|{conversation}|table|style=style_conv|show_all|selected={selected_row}|rebuild=True|>
<|part|class_name=card mt1|

<|layout|columns=1 50px|gap=20px|
<|
<|{current_user_message}|input|label=Write your message here...|on_action=send_message|class_name=fullwidth|change_delay=-1|>
|>
<|
<|⮞|button|on_action=send_message|change_delay=-1|>
|>
|>

<|
# 
|>
<|New Chat|button|id=reset_app_button|on_action=reset_chat|>
{: .text-left}

|>
|>
|>


|>
|>

<br/>


<|container|
-----------------------------------------------------------------
## Stock Price **Analysis**{: .color-primary} Dashboard

<step1|layout|columns=1 2|gap=20px|class_name=card|

<|
##### *STEP 1: Select a valid* *Ticker*{: .color-primary}
|>
<|
|>
<|
Please enter a valid ticker: 
|>
<ticker|
<|{selected_stock}|input|label=Stock|on_action=get_data_from_range|>
|ticker>

|step1>


<exp1|or Choose a Popular Ticker from the list|expandable|expanded=False|

<lout1|layout|columns=1 3|gap=10px|class_name=card|
<|
**Technology Sector  :**
|>
<|
<|{selected_stock}|toggle|lov=NVDA;MSFT;GOOG;AAPL;AMZN;META;AMD;INTC;AVGO|on_change=get_data_from_range|>
|>

<|
**Healthcare Sector  :**
|>
<|
<|{selected_stock}|toggle|lov=ABT;BIIB;BMRN;JNJ;PFE;MRK;MRNA;ISRG;UNH;CVS|on_change=get_data_from_range|>
|>

<|
**Automobile Sector  :**
|>
<|
<|{selected_stock}|toggle|lov=TSLA;RIVN;F;GM;RIDE;WKHS;NIO;GOEV;FUV;XPEV|on_change=get_data_from_range|>
|>

<|
**Financial Sector   :**
|>
<|
<|{selected_stock}|toggle|lov=BLK;JPM;PYPL;V;MA;C;MS;WFC;DFS;AMP|on_change=get_data_from_range|>
|>

<|
**Energy Sector      :**
|>
<|
<|{selected_stock}|toggle|lov=XOM;CVX;COP;SLB;HAL;EOG;DUK;NEE;D;KMI|on_change=get_data_from_range|>
|>

<|
**FMCG Sector        :**
|>
<|
<|{selected_stock}|toggle|lov=PG;KO;PEP;MDLZ;MNST;KHC;KDP;STZ;HAIN;KMB|on_change=get_data_from_range|>
|>

<|
**Entertainment Sector :**
|>
<|
<|{selected_stock}|toggle|lov=DIS;CMCSA;NFLX;EA;HAS;LYV;CNK|on_change=get_data_from_range|>
|>

|lout1>
|exp1>
|>

<br/>

<|container|
<step2|layout|columns=1 2|gap=20px|class_name=card|

<|
##### *STEP 2: Select* *Prediction years*{: .color-primary}
|>
<|
|>

<|
Selected number of prediction years: 
<|{n_years}|slider|min=1|max=5|>
|>
<|
<|{n_years}|input|label=Years|active=False|multiline=False|>
|>

<|PREDICT|button|on_action=forecast_display|class_name={'plain' if len(forecast)==0 else ''}|>
<|
|>

|step2>

<br/>

<|layout|columns=1|gap=10px|class_name=card mt1|
#### *Candlestick*{: .color-primary} *chart for* :  <|{selected_stock}|text|raw|> 
<|{data}|chart|type=candlestick|x=Date|open=Open|close=Close|low=Low|high=High|>
|>

|>

<|container|
<|Historical Data|expandable|expanded=False|
<|layout|columns=1 1|
<|
### Historical **closing**{: .color-primary} price
<|{data}|chart|mode=line|x=Date|y[1]=Open|y[2]=Close|>
|>

<|
### Historical **daily**{: .color-primary} trading volume
<|{data}|chart|mode=line|x=Date|y=Volume|>
|>
|>

### **Whole**{: .color-primary} historical data :  <|{selected_stock}|text|raw|>
<|{data}|table|>

|>

<br/>

<|container|
### **Forecast**{: .color-primary} Data

<|layout|columns=1 1|class_name=text-center|
<|  Pessimistic Forecast {int((forecast.loc[len(forecast)-1, 'Lower'] - forecast.loc[len(data), 'Lower'])/forecast.loc[len(data), 'Lower']*100)}%|text|class_name=h4 card|>

<|Optimistic Forecast {int((forecast.loc[len(forecast)-1, 'Upper'] - forecast.loc[len(data), 'Upper'])/forecast.loc[len(data), 'Upper']*100)}%|text|class_name=h4 card|>
|>


<|{forecast}|chart|mode=line|x=Date|y[1]=Lower|y[2]=Upper|>

<br/>


<|More info|button|on_action={lambda s: s.assign("show_dialog", True)}|>
{: .text-center}
|>

<br/>

<|
-----------------------------------------------------------------
<br/>
© 2023 - 2024. Made with ❤️ in India.
{: .text-center}
|>
<br/>

|>
"""



github_repo_md = """
<|container|
# GitHub ***Repository***{: .color-primary}

#### '**CAPITAL GAIN**{: .color-primary}' *project's GitHub repository [link provided below], is the central hub for our collaborative development efforts. It provides a comprehensive platform for managing and tracking the evolution of our software solution. We can find detailed documentation, version history, and codebase, fostering transparency and efficient collaboration among team members. With robust version control features, issue tracking, and pull request functionalities, our repository ensures structured and organized development processes, enhancing productivity and project management efficiency.*

<br/>

## https://github.com/hrajanna84/iisc-capstone-capitalgain

####Click [*Here*](https://github.com/hrajanna84/iisc-capstone-capitalgain) to visit the GitHub repository for more details.

<br/>
<br/>

-----------------------------------------------------------------
<br/>
© 2023 - 2024. Made with ❤️ in India.
{: .text-center}
|>
<br/>
"""


about_team_md = """
<|container|
# About ***Team***{: .color-primary}

#### *In the development of* '**CAPITAL GAIN**{: .color-primary}' *project, a cohesive team of skilled professionals collaborated seamlessly to bring our vision to life. Our team was composed of diverse individuals, each bringing unique expertise and perspectives to the table. From software engineers and designers to project managers and quality assurance specialists, every team member played a crucial role in the project's success. With clear communication channels and a shared commitment to excellence, we navigated challenges together, leveraging our collective strengths to overcome obstacles and deliver outstanding results. Each team member's dedication, creativity, and problem-solving abilities contributed to the project's development journey, fostering innovation and fostering a collaborative environment where ideas flourished. Through effective collaboration and mutual support, we not only achieved our project goals but also fostered a sense of camaraderie and achievement among team members.*

<|{image_path}|image|width="400px"|>

-----------------------------------------------------------------
<br/>
© 2023 - 2024. Made with ❤️ in India.
{: .text-center}
|>
<br/>
"""


menu_lov = [("Home-Page", Icon('images/icons/Home.svg', 'Home')),
            ('GitHub', Icon('images/icons/GitHub.svg', 'GitHub Repo')),
            ('About-Team', Icon('images/icons/TeamMembers.svg', 'About Team'))]

page = "Home-Page"


def menu_fct(state, var_name, var_value):
    """Function that is called when there is a change in the menu control."""
    state.page = var_value["args"][0]
    navigate(state, state.page.replace(" ", "-"))


ROOT = """
<|menu|label=Menu|lov={menu_lov}|on_action=menu_fct|>
"""

pages = {"/": ROOT,
         "Home-Page":root_page_md,
         'GitHub':github_repo_md,
         "About-Team":about_team_md
        }


tp_app = tp.Gui(pages=pages)
partial = tp_app.add_partial(partial_md)

# Initialize FinnHub agent
init_finnhub()

if __name__ == "__main__":
    if "OPENAI_API_KEY" in os.environ:
        api_key = os.environ["OPENAI_API_KEY"]
    elif len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        raise ValueError(
            "Please provide the OpenAI API key as an environment variable OPENAI_API_KEY or as a command line argument."
        )

    client = openai.Client(api_key=api_key)
    
    

    #Gui(page).run(debug=False, dark_mode=True, port=2452, use_reloader=False, title="💬 Taipy Chat")
    tp_app.run(debug=False, dark_mode=True, use_reloader=False, title="Capital Gain", favicon="./images/icons/CG_favicon.png")
