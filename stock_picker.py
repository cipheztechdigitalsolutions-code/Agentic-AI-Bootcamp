from dotenv import load_dotenv
import os

load_dotenv(override=True)


from crewai import Crew, Agent, Task, LLM, Process
from crewai.tools import tool
import yfinance as yf
import json


llm = LLM(
    model="openrouter/openai/gpt-5-nano",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("API_TOKEN")
)



@tool("Fetch Stock Data")                             
                                                       
def fetch_stock_data(ticker: str) -> str:              
    """
    Fetches real-time stock data for a given ticker symbol using Yahoo Finance.
    Returns price, P/E ratio, market cap, 52-week range, and analyst targets.
    Use this tool when you need current financial data for a specific stock.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT, NVDA, TSLA)
    """
  
    try:
        stock = yf.Ticker(ticker)                      
        info = stock.info                              

        if not info or "currentPrice" not in info:     
            return f"No data found for ticker: {ticker}. Please check the symbol."

        
        return f"""
        === {ticker.upper()} — {info.get('longName', 'N/A')} ===
        Current Price:    ${info.get('currentPrice', 'N/A')}
        P/E Ratio:        {info.get('trailingPE', 'N/A')}
        Forward P/E:      {info.get('forwardPE', 'N/A')}
        Market Cap:       ${info.get('marketCap', 0):,}
        52-Week Low:      ${info.get('fiftyTwoWeekLow', 'N/A')}
        52-Week High:     ${info.get('fiftyTwoWeekHigh', 'N/A')}
        Revenue Growth:   {info.get('revenueGrowth', 'N/A')}
        Profit Margin:    {info.get('profitMargins', 'N/A')}
        Debt/Equity:      {info.get('debtToEquity', 'N/A')}
        Free Cash Flow:   ${info.get('freeCashflow', 0):,}
        Analyst Target:   ${info.get('targetMeanPrice', 'N/A')}
        Recommendation:   {info.get('recommendationKey', 'N/A')}
        """
    except Exception as e:                             
        return f"Error fetching data for {ticker}: {str(e)}"



@tool("Search Market News")
def search_market_news(query: str) -> str:
        """
        Searches the web for the latest market news, trends, and analysis.
        Use this when you need current information about market conditions,
        sector trends, or company news that isn't in stock data.

        Args:
            query: Search query (e.g., 'technology sector outlook 2025', 'NVDA earnings analysis')
        """
        try:
            from duckduckgo_search import DDGS             

            with DDGS() as ddgs:                             
                results = list(ddgs.text(query, max_results=5))  

            if not results:
                return "No search results found. Try a different query."

            output = f"Search Results for: '{query}'\n{'='*50}\n"
            for i, r in enumerate(results, 1):               
                output += f"\n{i}. {r['title']}\n"           
                output += f"   {r['body']}\n"                 
                output += f"   Source: {r['href']}\n"         

            return output
        except Exception as e:
            return f"Search error: {str(e)}"


@tool("Screen Stocks")
def screen_stocks(sector: str) -> str:
    """
    Screens multiple stocks in a given sector and returns key metrics for comparison.
    Use this to identify the best candidates in a sector before deep analysis.

    Args:
        sector: Market sector to screen (e.g., 'technology', 'healthcare', 'energy', 'finance')
    """
    
    sector_stocks = {
        "technology": ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AVGO", "CRM", "PLTR", "CRWD", "AMD"],
        "healthcare": ["LLY", "UNH", "JNJ", "PFE", "ABBV", "MRK", "TMO", "ABT", "AMGN", "ISRG"],
        "energy": ["XOM", "CVX", "NEE", "FSLR", "ENPH", "SLB", "COP", "EOG", "PSX", "OXY"],
        "finance": ["JPM", "V", "MA", "BAC", "GS", "MS", "BLK", "SCHW", "AXP", "SPGI"],
    }

    tickers = sector_stocks.get(sector.lower(), sector_stocks["technology"]) 

    output = f"Stock Screening Results — {sector.upper()} Sector\n{'='*60}\n"

    for ticker in tickers[:8]:                           
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            output += f"\n{ticker} — {info.get('longName', 'N/A')}\n"
            output += f"  Price: ${info.get('currentPrice', 'N/A')}"
            output += f"  | P/E: {info.get('trailingPE', 'N/A')}"
            output += f"  | Growth: {info.get('revenueGrowth', 'N/A')}"
            output += f"  | Margin: {info.get('profitMargins', 'N/A')}"
            output += f"  | Target: ${info.get('targetMeanPrice', 'N/A')}\n"
        except Exception:
            output += f"\n{ticker} — Data unavailable\n"

    return output



@tool("Analyze Company")
def analyze_company(ticker: str) -> str:
    """
    Performs a deep fundamental analysis of a specific company.
    Returns detailed financials, competitive position, and risk factors.
    Use this for in-depth analysis of your top stock picks.

    Args:
        ticker: Stock ticker symbol (e.g., NVDA, AAPL, MSFT)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        
        hist = stock.history(period="6mo")               
        price_6m_ago = hist['Close'].iloc[0] if len(hist) > 0 else None   
        price_now = hist['Close'].iloc[-1] if len(hist) > 0 else None    

        
        if price_6m_ago and price_now:
            six_month_return = ((price_now - price_6m_ago) / price_6m_ago) * 100
            momentum = f"{six_month_return:.1f}%"
        else:
            momentum = "N/A"

        return f"""
        ══════════════════════════════════════════════════
        DEEP ANALYSIS: {ticker.upper()} — {info.get('longName', 'N/A')}
        ══════════════════════════════════════════════════

        PRICE & VALUATION
        ─────────────────
        Current Price:     ${info.get('currentPrice', 'N/A')}
        P/E (Trailing):    {info.get('trailingPE', 'N/A')}
        P/E (Forward):     {info.get('forwardPE', 'N/A')}
        PEG Ratio:         {info.get('pegRatio', 'N/A')}
        Price/Book:        {info.get('priceToBook', 'N/A')}
        6-Month Return:    {momentum}

        FINANCIALS
        ──────────
        Revenue (TTM):     ${info.get('totalRevenue', 0):,}
        Net Income:        ${info.get('netIncomeToCommon', 0):,}
        Revenue Growth:    {info.get('revenueGrowth', 'N/A')}
        Profit Margin:     {info.get('profitMargins', 'N/A')}
        Operating Margin:  {info.get('operatingMargins', 'N/A')}

        FINANCIAL HEALTH
        ────────────────
        Debt/Equity:       {info.get('debtToEquity', 'N/A')}
        Current Ratio:     {info.get('currentRatio', 'N/A')}
        Free Cash Flow:    ${info.get('freeCashflow', 0):,}

        ANALYST CONSENSUS
        ─────────────────
        Recommendation:    {info.get('recommendationKey', 'N/A')}
        Mean Target:       ${info.get('targetMeanPrice', 'N/A')}
        Low Target:        ${info.get('targetLowPrice', 'N/A')}
        High Target:       ${info.get('targetHighPrice', 'N/A')}
        Number of Analysts:{info.get('numberOfAnalystOpinions', 'N/A')}

        BUSINESS OVERVIEW
        ─────────────────
        Sector:            {info.get('sector', 'N/A')}
        Industry:          {info.get('industry', 'N/A')}
        Employees:         {info.get('fullTimeEmployees', 'N/A')}
        Summary:           {str(info.get('longBusinessSummary', 'N/A'))[:300]}...
        """
    except Exception as e:
        return f"Error analyzing {ticker}: {str(e)}"



market_researcher = Agent(
    role="Senior Market Research Analyst",              
    goal="Analyze current market conditions and identify the most promising sectors and trends for investment",
                                                        
    backstory="""You are a veteran market analyst with 20 years at JP Morgan.
    You have a keen eye for macroeconomic trends and sector rotation. You always
    start with the big picture before drilling into specifics. Your analysis
    has helped clients avoid major downturns and capitalize on bull runs.
    You always support your analysis with data and never make claims without evidence.""",
                                                        
                                                        
                                                        
    llm=llm,                                           
    tools=[search_market_news, fetch_stock_data],       
    verbose=True,                                       
    max_iter=15                                        
)


stock_screener = Agent(
    role="Quantitative Stock Screener",
    goal="Screen and identify the most promising stock opportunities using data-driven analysis",
    backstory="""You are a quant analyst who built screening algorithms at
    Renaissance Technologies. You combine fundamental metrics (P/E, growth,
    margins) with momentum data to find hidden gems. You never recommend a
    stock without solid data backing. You always rank picks by conviction level.""",
    llm=llm,
    tools=[screen_stocks, analyze_company, fetch_stock_data],  
    verbose=True,
    max_iter=20                                        
)



investment_advisor = Agent(
    role="Chief Investment Strategist",
    goal="Generate actionable investment recommendations with clear entry points, targets, and risk management",
    backstory="""You are the CIO of a boutique investment firm managing $500M in AUM.
    You are known for clear, actionable advice that balances risk and reward.
    You always include specific entry prices, 12-month price targets, and stop-loss
    levels. You never make a recommendation without considering what could go wrong.
    Your reports are used by professional investors to make real allocation decisions.""",
    llm=llm,
    tools=[analyze_company, fetch_stock_data],          
    verbose=True,
    max_iter=15
)


market_research_task = Task(
    description="""Analyze the current market conditions for the {sector} sector.

    You MUST use the Search Market News tool to find recent news and trends.
    You MUST use the Fetch Stock Data tool on at least 2 major stocks in this sector.

    Your analysis should cover:
    1. Overall sector trend (bullish / bearish / neutral) and why
    2. Key drivers and catalysts pushing the sector
    3. Major risks and headwinds to watch for
    4. Recent news or events affecting the sector
    5. Your assessment: Is now a good time to invest in this sector?""",
                                                        
                                                        

    expected_output="""A comprehensive market analysis report including:
    - Sector trend assessment with supporting data
    - At least 3 key catalysts/drivers
    - At least 3 risk factors
    - Recent relevant news
    - Clear investment recommendation for the sector""",
                                                        

    agent=market_researcher                              
)

screening_task = Task(
    description="""Based on the market research provided, screen for the best
    stock opportunities in the {sector} sector.

    You MUST use the Screen Stocks tool to get data on stocks in this sector.
    Then use the Analyze Company tool to deep-dive the top 3-4 most promising picks.

    Rank your picks by conviction (highest first). For each pick, explain:
    1. Why this stock stands out (competitive advantages)
    2. Key financial metrics that support the pick
    3. What could go wrong (risks specific to this company)
    4. How it fits with the current sector outlook from the research""",

    expected_output="""A ranked list of top 3 stock picks with:
    - Ticker, company name, current price
    - Key metrics (P/E, growth, margins, analyst target)
    - Investment thesis (why this stock)
    - Specific risks for each pick
    - Conviction level (High / Medium / Low)""",

    agent=stock_screener,
    context=[market_research_task]                       
                                                         
)


recommendation_task = Task(
    description="""Based on the market analysis and stock screening results,
    create a professional investment recommendation report.

    Use the Analyze Company tool to verify data on the final picks if needed.

    For each recommended stock, you MUST include:
    1. Investment thesis — why buy this stock (2-3 sentences)
    2. Entry price — suggested buy price or range
    3. 12-month price target — where you think it goes
    4. Stop-loss level — where to cut losses (typically 10-15% below entry)
    5. Risk/reward ratio
    6. Key risks — what could make this pick fail
    7. Portfolio allocation — suggested % of portfolio (should sum to 100% of allocated capital)

    End with an overall portfolio strategy summary and important disclaimers.""",

    expected_output="""A professional investment report formatted as:

    STOCK PICKER AGENT — INVESTMENT RECOMMENDATIONS
    Sector: {sector}
    Date: [current date]
    ================================================

    MARKET OVERVIEW
    [Brief sector summary]

    PICK #1: [TICKER] — [Company Name] (Conviction: High/Medium)
    - Thesis: ...
    - Entry: $XX | Target: $XX | Stop-Loss: $XX
    - Risk/Reward: X:1
    - Allocation: XX%
    - Key Risks: ...

    [Repeat for each pick]

    PORTFOLIO STRATEGY SUMMARY
    [Overall allocation and risk management approach]

    DISCLAIMER
    [Standard investment disclaimer]""",

    agent=investment_advisor,
    context=[market_research_task, screening_task],      
    output_file="output/stock_recommendations.md"         
)


stock_picker_crew = Crew(
    agents=[market_researcher, stock_screener, investment_advisor],
                                                       
    tasks=[market_research_task, screening_task, recommendation_task],
                                                        
    process=Process.sequential,                          
                                                        
    verbose=True                                        
)


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)                 

    
    sector = input("Enter sector to analyze (technology/healthcare/energy/finance): ").strip()
    if not sector:
        sector = "technology"                           

    print(f"\n{'='*60}")
    print(f"  STOCK PICKER AGENT — Analyzing: {sector.upper()}")
    print(f"{'='*60}\n")

   
    result = stock_picker_crew.kickoff(
        inputs={"sector": sector}                         
    )

   
    print(f"\n{'='*60}")
    print("  FINAL INVESTMENT RECOMMENDATIONS")
    print(f"{'='*60}")
    print(result.raw)                                     

   
    print(f"\n--- Token Usage ---")
    print(f"Total tokens: {result.token_usage}")
