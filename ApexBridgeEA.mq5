#property strict
#property version   "1.00"
#property description "Apex MT5 bridge EA: orders/status/positions/market JSON sync"

input int BridgePollSeconds = 2;
input int ExportBars = 300;
input string SymbolsCsv = "EURUSD,GBPUSD,USDJPY,AUDUSD,XAUUSD";
input string BridgeDir = "bridge";

string BridgeFile(const string fileName)
{
   return BridgeDir + "/" + fileName;
}

string MarketDir()
{
   return BridgeDir + "/market";
}

string MarketFile(const string fileName)
{
   return MarketDir() + "/" + fileName;
}

int OnInit()
{
   EnsureBridgeFolders();
   WriteJsonFile(BridgeFile("orders.json"), "[]");
   WriteStatus();
   WritePositions();
   ExportMarketData();
   EventSetTimer(MathMax(1, BridgePollSeconds));
   Print("ApexBridgeEA initialized.");
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("ApexBridgeEA deinitialized. reason=", reason);
}

void OnTick()
{
}

void OnTimer()
{
   EnsureBridgeFolders();
   ProcessOrders();
   WriteStatus();
   WritePositions();
   ExportMarketData();
}

void EnsureBridgeFolders()
{
   FolderCreate(BridgeDir);
   FolderCreate(MarketDir());
}

int FileFlags()
{
   return FILE_TXT | FILE_ANSI;
}

bool ReadTextFile(const string path, string &content)
{
   content = "";
   int handle = FileOpen(path, FileFlags() | FILE_READ);
   if(handle == INVALID_HANDLE)
      return false;

   while(!FileIsEnding(handle))
      content += FileReadString(handle);

   FileClose(handle);
   return true;
}

bool WriteJsonFile(const string path, const string payload)
{
   int handle = FileOpen(path, FileFlags() | FILE_WRITE);
   if(handle == INVALID_HANDLE)
   {
      Print("FileOpen write failed: ", path, " error=", GetLastError());
      return false;
   }
   FileWriteString(handle, payload);
   FileClose(handle);
   return true;
}

string JsonEscape(string value)
{
   StringReplace(value, "\\", "\\\\");
   StringReplace(value, "\"", "\\\"");
   StringReplace(value, "\r", "");
   StringReplace(value, "\n", "\\n");
   StringReplace(value, "\t", "\\t");
   return value;
}

string NumToJson(double value, int digits = 8)
{
   string out = DoubleToString(value, digits);
   while(StringLen(out) > 1 && StringSubstr(out, StringLen(out) - 1, 1) == "0")
      out = StringSubstr(out, 0, StringLen(out) - 1);
   if(StringLen(out) > 0 && StringSubstr(out, StringLen(out) - 1, 1) == ".")
      out = StringSubstr(out, 0, StringLen(out) - 1);
   if(out == "-0")
      out = "0";
   return out;
}

string TrimString(string value)
{
   StringTrimLeft(value);
   StringTrimRight(value);
   return value;
}

bool JsonExtractObjectByKey(const string src, const string key, int startPos, string &obj, int &nextPos)
{
   string needle = "\"" + key + "\"";
   int pos = StringFind(src, needle, startPos);
   if(pos < 0)
      return false;

   int colon = StringFind(src, ":", pos + StringLen(needle));
   if(colon < 0)
      return false;

   int firstBrace = StringFind(src, "{", colon + 1);
   if(firstBrace < 0)
      return false;

   int depth = 0;
   bool inString = false;
   bool escaped = false;
   int len = StringLen(src);

   for(int i = firstBrace; i < len; i++)
   {
      string ch = StringSubstr(src, i, 1);

      if(inString)
      {
         if(escaped)
         {
            escaped = false;
            continue;
         }
         if(ch == "\\")
            escaped = true;
         else if(ch == "\"")
            inString = false;
         continue;
      }

      if(ch == "\"")
      {
         inString = true;
         continue;
      }

      if(ch == "{")
         depth++;
      else if(ch == "}")
      {
         depth--;
         if(depth == 0)
         {
            obj = StringSubstr(src, firstBrace, i - firstBrace + 1);
            nextPos = i + 1;
            return true;
         }
      }
   }

   return false;
}

bool JsonExtractString(const string src, const string key, string &out)
{
   out = "";
   string needle = "\"" + key + "\"";
   int pos = StringFind(src, needle);
   if(pos < 0)
      return false;

   int colon = StringFind(src, ":", pos + StringLen(needle));
   if(colon < 0)
      return false;

   int start = StringFind(src, "\"", colon + 1);
   if(start < 0)
      return false;

   int len = StringLen(src);
   bool escaped = false;
   for(int i = start + 1; i < len; i++)
   {
      string ch = StringSubstr(src, i, 1);
      if(escaped)
      {
         escaped = false;
         continue;
      }
      if(ch == "\\")
      {
         escaped = true;
         continue;
      }
      if(ch == "\"")
      {
         out = StringSubstr(src, start + 1, i - start - 1);
         return true;
      }
   }
   return false;
}

bool JsonExtractNumber(const string src, const string key, double &out)
{
   string needle = "\"" + key + "\"";
   int pos = StringFind(src, needle);
   if(pos < 0)
      return false;

   int colon = StringFind(src, ":", pos + StringLen(needle));
   if(colon < 0)
      return false;

   int i = colon + 1;
   int len = StringLen(src);
   while(i < len)
   {
      string ch = StringSubstr(src, i, 1);
      if(ch != " " && ch != "\t" && ch != "\r" && ch != "\n")
         break;
      i++;
   }

   int start = i;
   while(i < len)
   {
      string ch = StringSubstr(src, i, 1);
      bool isNum = (ch == "-" || ch == "+" || ch == "." || (ch >= "0" && ch <= "9") || ch == "e" || ch == "E");
      if(!isNum)
         break;
      i++;
   }

   if(i <= start)
      return false;

   string token = StringSubstr(src, start, i - start);
   out = StringToDouble(token);
   return true;
}

bool JsonExtractInteger(const string src, const string key, long &out)
{
   double val;
   if(!JsonExtractNumber(src, key, val))
      return false;
   out = (long)MathRound(val);
   return true;
}

bool ExecuteOrderRequest(const string reqJson)
{
   string symbol;
   double volume = 0.0;
   long orderType = ORDER_TYPE_BUY;
   long action = TRADE_ACTION_DEAL;
   long deviation = 20;
   long magic = 0;
   long typeTime = ORDER_TIME_GTC;
   long typeFilling = ORDER_FILLING_IOC;
   long positionTicket = 0;
   string comment = "APEX bridge";
   double price = 0.0;

   if(!JsonExtractString(reqJson, "symbol", symbol))
   {
      Print("Order skipped: missing symbol in request");
      return false;
   }

   if(!JsonExtractNumber(reqJson, "volume", volume) || volume <= 0.0)
   {
      Print("Order skipped: invalid volume for symbol ", symbol);
      return false;
   }

   JsonExtractInteger(reqJson, "type", orderType);
   JsonExtractInteger(reqJson, "action", action);
   JsonExtractInteger(reqJson, "deviation", deviation);
   JsonExtractInteger(reqJson, "magic", magic);
   JsonExtractInteger(reqJson, "type_time", typeTime);
   JsonExtractInteger(reqJson, "type_filling", typeFilling);
   JsonExtractInteger(reqJson, "position", positionTicket);
   JsonExtractString(reqJson, "comment", comment);
   JsonExtractNumber(reqJson, "price", price);

   SymbolSelect(symbol, true);

   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick))
   {
      Print("Order skipped: SymbolInfoTick failed for ", symbol, " error=", GetLastError());
      return false;
   }

   if(price <= 0.0)
   {
      if(orderType == ORDER_TYPE_BUY)
         price = tick.ask;
      else
         price = tick.bid;
   }

   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = (ENUM_TRADE_REQUEST_ACTIONS)action;
   request.symbol = symbol;
   request.volume = volume;
   request.type = (ENUM_ORDER_TYPE)orderType;
   request.price = price;
   request.deviation = (ulong)MathMax(0, deviation);
   request.magic = magic;
   request.comment = comment;
   request.type_time = (ENUM_ORDER_TYPE_TIME)typeTime;
   request.type_filling = (ENUM_ORDER_TYPE_FILLING)typeFilling;

   if(positionTicket > 0)
      request.position = (ulong)positionTicket;

   bool ok = OrderSend(request, result);
   if(!ok)
   {
      Print("OrderSend failed symbol=", symbol, " retcode=", result.retcode, " error=", GetLastError());
      return false;
   }

   Print("Order executed symbol=", symbol, " retcode=", result.retcode, " order=", result.order, " deal=", result.deal);
   return true;
}

void ProcessOrders()
{
   string raw;
   if(!ReadTextFile(BridgeFile("orders.json"), raw))
   {
      WriteJsonFile(BridgeFile("orders.json"), "[]");
      return;
   }

   string trimmed = TrimString(raw);
   if(trimmed == "" || trimmed == "[]")
      return;

   int cursor = 0;
   int processed = 0;
   string reqObj;
   int nextPos = 0;

   while(JsonExtractObjectByKey(trimmed, "request", cursor, reqObj, nextPos))
   {
      ExecuteOrderRequest(reqObj);
      processed++;
      cursor = nextPos;
   }

   if(processed > 0)
      Print("Processed bridge orders: ", processed);

   WriteJsonFile(BridgeFile("orders.json"), "[]");
}

string BuildStatusJson()
{
   long login = (long)AccountInfoInteger(ACCOUNT_LOGIN);
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double margin = AccountInfoDouble(ACCOUNT_MARGIN);
   double marginFree = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double marginLevel = AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);
   string currency = AccountInfoString(ACCOUNT_CURRENCY);
   string server = AccountInfoString(ACCOUNT_SERVER);
   string company = AccountInfoString(ACCOUNT_COMPANY);

   string json = "{";
   json += "\"login\":" + IntegerToString((int)login) + ",";
   json += "\"balance\":" + NumToJson(balance, 2) + ",";
   json += "\"equity\":" + NumToJson(equity, 2) + ",";
   json += "\"margin\":" + NumToJson(margin, 2) + ",";
   json += "\"margin_free\":" + NumToJson(marginFree, 2) + ",";
   json += "\"margin_level\":" + NumToJson(marginLevel, 2) + ",";
   json += "\"currency\":\"" + JsonEscape(currency) + "\",";
   json += "\"server\":\"" + JsonEscape(server) + "\",";
   json += "\"company\":\"" + JsonEscape(company) + "\",";
   json += "\"timestamp\":" + IntegerToString((int)TimeCurrent());
   json += "}";
   return json;
}

void WriteStatus()
{
   WriteJsonFile(BridgeFile("status.json"), BuildStatusJson());
}

string BuildPositionsJson()
{
   int total = PositionsTotal();
   string json = "[";

   for(int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;

      if(StringLen(json) > 1)
         json += ",";

      string symbol = PositionGetString(POSITION_SYMBOL);
      long type = PositionGetInteger(POSITION_TYPE);
      long magic = PositionGetInteger(POSITION_MAGIC);
      long timeOpen = PositionGetInteger(POSITION_TIME);
      double volume = PositionGetDouble(POSITION_VOLUME);
      double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      double sl = PositionGetDouble(POSITION_SL);
      double tp = PositionGetDouble(POSITION_TP);
      double profit = PositionGetDouble(POSITION_PROFIT);
      double swap = PositionGetDouble(POSITION_SWAP);

      json += "{";
      json += "\"ticket\":" + (string)ticket + ",";
      json += "\"symbol\":\"" + JsonEscape(symbol) + "\",";
      json += "\"type\":" + IntegerToString((int)type) + ",";
      json += "\"magic\":" + IntegerToString((int)magic) + ",";
      json += "\"time\":" + IntegerToString((int)timeOpen) + ",";
      json += "\"volume\":" + NumToJson(volume, 2) + ",";
      json += "\"price_open\":" + NumToJson(openPrice, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)) + ",";
      json += "\"price_current\":" + NumToJson(currentPrice, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)) + ",";
      json += "\"sl\":" + NumToJson(sl, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)) + ",";
      json += "\"tp\":" + NumToJson(tp, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)) + ",";
      json += "\"profit\":" + NumToJson(profit, 2) + ",";
      json += "\"swap\":" + NumToJson(swap, 2);
      json += "}";
   }

   json += "]";
   return json;
}

void WritePositions()
{
   WriteJsonFile(BridgeFile("positions.json"), BuildPositionsJson());
}

string BuildTickJson(const MqlTick &tick)
{
   string json = "{";
   json += "\"time\":" + IntegerToString((int)tick.time) + ",";
   json += "\"bid\":" + NumToJson(tick.bid, 8) + ",";
   json += "\"ask\":" + NumToJson(tick.ask, 8) + ",";
   json += "\"last\":" + NumToJson(tick.last, 8) + ",";
   json += "\"volume\":" + IntegerToString((int)tick.volume) + ",";
   json += "\"time_msc\":" + (string)tick.time_msc;
   json += "}";
   return json;
}

string BuildSymbolInfoJson(const string symbol)
{
   long fillingMode = 0;
   if(!SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE, fillingMode))
      fillingMode = ORDER_FILLING_IOC;

   string json = "{";
   json += "\"symbol\":\"" + JsonEscape(symbol) + "\",";
   json += "\"filling_mode\":" + IntegerToString((int)fillingMode) + ",";
   json += "\"digits\":" + IntegerToString((int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)) + ",";
   json += "\"point\":" + NumToJson(SymbolInfoDouble(symbol, SYMBOL_POINT), 10) + ",";
   json += "\"timestamp\":" + IntegerToString((int)TimeCurrent());
   json += "}";
   return json;
}

string BuildRatesJson(MqlRates &rates[])
{
   string json = "[";
   int total = ArraySize(rates);
   for(int i = 0; i < total; i++)
   {
      if(i > 0)
         json += ",";

      json += "{";
      json += "\"time\":" + IntegerToString((int)rates[i].time) + ",";
      json += "\"open\":" + NumToJson(rates[i].open, 8) + ",";
      json += "\"high\":" + NumToJson(rates[i].high, 8) + ",";
      json += "\"low\":" + NumToJson(rates[i].low, 8) + ",";
      json += "\"close\":" + NumToJson(rates[i].close, 8) + ",";
      json += "\"tick_volume\":" + IntegerToString((int)rates[i].tick_volume) + ",";
      json += "\"spread\":" + IntegerToString((int)rates[i].spread) + ",";
      json += "\"real_volume\":" + IntegerToString((int)rates[i].real_volume);
      json += "}";
   }

   json += "]";
   return json;
}

void ExportRatesForSymbol(const string symbol, ENUM_TIMEFRAMES timeframe, const string fileSuffix)
{
   MqlRates rates[];
   int copied = CopyRates(symbol, timeframe, 1, ExportBars, rates);
   if(copied <= 0)
      return;

   ArraySetAsSeries(rates, false);
   WriteJsonFile(MarketFile(symbol + "_" + fileSuffix + ".json"), BuildRatesJson(rates));
}

void ExportMarketData()
{
   string symbols[];
   int count = StringSplit(SymbolsCsv, ',', symbols);
   if(count <= 0)
      return;

   for(int i = 0; i < count; i++)
   {
      string symbol = TrimString(symbols[i]);
      if(symbol == "")
         continue;

      if(!SymbolSelect(symbol, true))
         continue;

      MqlTick tick;
      if(SymbolInfoTick(symbol, tick))
         WriteJsonFile(MarketFile(symbol + "_tick.json"), BuildTickJson(tick));

      WriteJsonFile(MarketFile(symbol + "_info.json"), BuildSymbolInfoJson(symbol));
      ExportRatesForSymbol(symbol, PERIOD_M5, "5");
      ExportRatesForSymbol(symbol, PERIOD_M15, "15");
      ExportRatesForSymbol(symbol, PERIOD_H1, "60");
      ExportRatesForSymbol(symbol, PERIOD_H4, "240");
   }
}
