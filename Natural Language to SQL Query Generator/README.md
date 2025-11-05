# Natural Language to SQL Query System

## 🎯 Business Objective

This project enables non-technical users to query databases using natural language instead of SQL. By leveraging Large Language Models (LLMs), users can ask questions in plain English and receive accurate data insights without needing to know SQL syntax. This democratizes data access across organizations, reducing the dependency on data analysts for routine queries and accelerating decision-making processes.

### Key Benefits:
- **Accessibility**: Anyone can query databases without SQL knowledge
- **Efficiency**: Instant query generation and execution
- **Safety**: Built-in validation prevents destructive operations
- **Transparency**: Generated SQL is displayed for review and learning

---

### AI/ML Framework:
- **LangChain**: Orchestration framework for LLM workflows
- **Ollama**: Local LLM runtime (using Llama 3.2 model)
- **LangChain Ollama Integration**: Seamless LLM connectivity

### Key Features:
1. **Schema Introspection**: Automatically reads database structure
2. **Prompt Engineering**: Custom-designed prompts for accurate SQL generation
3. **Safety Validation**: Regex-based checks to prevent dangerous operations
4. **Query Optimization**: Automatic LIMIT clauses to prevent large result sets
5. **Interactive Querying**: Natural language interface with DataFrame outputs

---

## 📊 System Architecture

The system follows a multi-stage pipeline:

1. **Database Connection**: Establishes connection to MySQL database using SQLAlchemy
2. **Schema Extraction**: Introspects database to get all tables and columns
3. **Natural Language Input**: User asks questions in plain English
4. **LLM Processing**: Ollama (Llama 3.2) generates SQL from the question + schema context
5. **Safety Validation**: Checks for dangerous operations and multi-statement queries
6. **Query Execution**: Runs validated SQL against the database
7. **Result Display**: Returns results as formatted Pandas DataFrame

---

## 🔒 Safety Mechanisms

The system includes multiple layers of protection:

- **Operation Filtering**: Blocks INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, CREATE, GRANT, REVOKE
- **Multi-Statement Prevention**: Prevents SQL injection via stacked queries
- **Automatic Limits**: Adds LIMIT clauses to prevent accidental large data pulls
- **Read-Only Queries**: Only SELECT operations are permitted

---

## 📈 Results & Demonstrations

### Example Query: Customer Order Count

**Natural Language Question:**
```
"How many orders did customer number 363 make?"
```

**Generated SQL:**
```sql
SELECT COUNT(DISTINCT o.orderNumber) AS orderCount 
FROM orders o 
JOIN customers c ON o.customerNumber = c.customerNumber 
WHERE c.customerNumber = 363 
LIMIT 50
```

**Result:**

| orderCount |
|------------|
| 3          |

**Rows returned:** 1

---

### Database Schema Overview

The system successfully connected to the database and extracted the complete schema including:

**Tables Analyzed:**
- `customers` - Customer information with 13 columns (customerNumber, customerName, contact details, location, creditLimit, etc.)
- `employees` - Employee records with role and reporting structure
- `offices` - Office locations and contact information
- `orders` - Order transactions with dates and status
- `orderdetails` - Line items for each order with pricing
- `products` - Product catalog with inventory details
- `productlines` - Product categorization
- `payments` - Payment records and transaction history

---

## 🚀 Use Cases

This system is ideal for:

- **Business Analysts**: Quick ad-hoc data exploration
- **Product Managers**: Understanding customer behavior and trends
- **Executives**: Self-service reporting and dashboards
- **Customer Support**: Looking up order and customer information
- **Sales Teams**: Analyzing sales performance and metrics

