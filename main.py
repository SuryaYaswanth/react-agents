from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

def main():

    information = """

        Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person in the world since 2025; as of February 2026, Forbes estimates his net worth to be around US$852 billion.

        Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.

        In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research, but later left; growing discontent with the organization's direction and their leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired the social network Twitter, implementing significant changes, and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017. In November 2025, a Tesla pay package worth $1 trillion for Musk was approved, which he is to receive over 10 years if he meets specific goals.

        Musk was the largest donor in the 2024 U.S. presidential election, where he supported Donald Trump. After Trump was inaugurated as president in early 2025, Musk served as Senior Advisor to the President and as the de facto head of the Department of Government Efficiency (DOGE). After a public feud with Trump, Musk left the Trump administration and returned to managing his companies. Musk is a supporter of global far-right figures, causes, and political parties. His political activities, views, and statements have made him a polarizing figure. Musk has been criticized for COVID-19 misinformation, promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service, following his pledge to decrease censorship. His role in the second Trump administration attracted public backlash, particularly in response to DOGE. The emails he sent to Jeffrey Epstein are included in the Epstein files, which were published between 2025–26 and became a topic of worldwide debate.

        Early life
        See also: Musk family
        Elon Reeve Musk was born on June 28, 1971, in Pretoria, South Africa's administrative capital.[1][2] He is of British and Pennsylvania Dutch ancestry.[3][4] His mother, Maye (née Haldeman), is a model and dietitian born in Saskatchewan, Canada, and raised in South Africa.[5][6][7] Musk therefore holds both South African and Canadian citizenship from birth.[8] His father, Errol Musk, is a South African electromechanical engineer, pilot, sailor, consultant, emerald dealer, and property developer, who partly owned a rental lodge at Timbavati Private Nature Reserve.[9][10][11][12]

        His maternal grandfather, Joshua N. Haldeman, who died in a plane crash when Elon was a toddler, was an American-born Canadian chiropractor, aviator and political activist in the technocracy movement[13][14] who moved to South Africa in 1950.[15]

        Elon has a younger brother, Kimbal, a younger sister, Tosca, and four paternal half-siblings.[16][17][7][18] Musk was baptized as a child in the Anglican Church of Southern Africa.[19][20] Despite both Elon and Errol previously stating that Errol was a part owner of a Zambian emerald mine,[12] in 2023, Errol recounted that the deal he made was to receive "a portion of the emeralds produced at three small mines".[21][22] Errol was elected to the Pretoria City Council as a representative of the anti-apartheid Progressive Party and has said that his children shared their father's dislike of apartheid.[1]
        """
    
    summary_template = """
        given the information {information} about a person, I want you to summarize the information in a few sentences, and then answer the question "What is the person's name?".
        1. Summarize the information in a few sentences.
        2. Two interesting facts about the person.
        """
    summary_prompt_template = PromptTemplate(input_variables=["information"], 
                                             template=summary_template)
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)
    print("Hello from react-agents!")


if __name__ == "__main__":
    main()
