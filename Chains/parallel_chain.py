from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

model2 = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

# 1st prompt -> detailed report
prompt1 = PromptTemplate(
    template= "generate a short and simple notes on the {text}",
    input_variables=["text"]
)

# 2nd prompt -> key points
prompt2 = PromptTemplate(
    template= "generate 5 short question answers from the following text: {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template= "merge the provided notes and quiz into a single document: \n notes -> {notes} \n quiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes" : prompt1 | model1 | parser,
    "quiz" : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = '''A method is a block of code which only runs when it is called.

You can pass data, known as parameters, into a method.

Methods are used to perform certain actions, and they are also known as functions.

Why use methods? To reuse code: define the code once, and use it many times.

Create a Method
A method must be declared within a class. It is defined with the name of the method, followed by parentheses (). Java provides some pre-defined methods, such as System.out.println(), but you can also create your own methods to perform certain actions:

ExampleGet your own Java Server
Create a method inside Main:

public class Main {
  static void myMethod() {
    // code to be executed
  }
}
 
Example Explained
myMethod() is the name of the method
static means that the method belongs to the Main class and not an object of the Main class. You will learn more about objects and how to access methods through objects later in this tutorial.
void means that this method does not have a return value. You will learn more about return values later in this chapter
Call a Method
To call a method in Java, write the method's name followed by two parentheses () and a semicolon;

In the following example, myMethod() is used to print a text (the action), when it is called:

Example
Inside main, call the myMethod() method:

public class Main {
  static void myMethod() {
    System.out.println("I just got executed!");
  }

  public static void main(String[] args) {
    myMethod();
  }
}

// Outputs "I just got executed!"
 

A method can also be called multiple times:

Example
public class Main {
  static void myMethod() {
    System.out.println("I just got executed!");
  }

  public static void main(String[] args) {
    myMethod();
    myMethod();
    myMethod();
  }
}

// I just got executed!
// I just got executed!
// I just got executed!

'''

result = chain.invoke({"text": text})

print("Merged Result:", result)