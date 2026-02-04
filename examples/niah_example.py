"""
Needle-in-a-Haystack (NIAH) example for RLM.

This example generates a synthetic haystack with hidden needles
and uses RLM to find specific information.
"""

import os
import sys
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlm import RLM, RLMConfig


def generate_haystack(
    num_paragraphs: int = 100,
    needle_position: int = 42,
    secret_code: str = "PHOENIX-7749"
) -> str:
    """
    Generate a haystack of random text with a hidden needle.
    
    Args:
        num_paragraphs: Number of paragraphs to generate
        needle_position: Paragraph index where the needle is hidden
        secret_code: The secret code to hide
        
    Returns:
        The generated text with the hidden needle
    """
    # Template paragraphs (filler content)
    templates = [
        "The weather today is particularly {adj}. Many people are {action} in the {place}. "
        "Scientists have observed that {observation}. This has led to {consequence}.",
        
        "In recent news, {organization} announced their plans for {project}. "
        "The initiative will involve {number} participants from {location}. "
        "Expected completion is set for {timeframe}.",
        
        "A new study from {university} reveals interesting findings about {topic}. "
        "Researchers found that {finding}. This could have implications for {field}.",
        
        "The local {establishment} has introduced a new {product}. "
        "Customers have been {reaction} about the changes. "
        "The {manager} stated that {quote}.",
        
        "Historical records indicate that {event} occurred in {year}. "
        "This was a significant moment for {group}. "
        "Many historians believe {interpretation}.",
    ]
    
    adjectives = ["pleasant", "unusual", "remarkable", "ordinary", "unexpected"]
    actions = ["walking", "gathering", "celebrating", "working", "relaxing"]
    places = ["park", "city center", "community hall", "marketplace", "library"]
    observations = ["bird migration patterns have changed", "temperatures are rising", 
                   "plant growth is accelerating", "wildlife behavior is shifting"]
    consequences = ["new research opportunities", "policy discussions", 
                   "public awareness campaigns", "scientific debates"]
    organizations = ["TechCorp", "GlobalUnited", "FutureVentures", "InnovateLabs"]
    projects = ["expansion", "modernization", "collaboration", "sustainability"]
    universities = ["MIT", "Stanford", "Oxford", "Cambridge", "Harvard"]
    topics = ["human behavior", "climate patterns", "economic trends", "social media impact"]
    findings = ["correlation is stronger than expected", "previous assumptions were incorrect",
               "new patterns have emerged", "data supports the hypothesis"]
    fields = ["medicine", "education", "technology", "environment", "psychology"]
    
    paragraphs = []
    
    for i in range(num_paragraphs):
        if i == needle_position:
            # Insert the needle
            needle = f"""
IMPORTANT CLASSIFIED INFORMATION - PARAGRAPH {i}
The secret access code for Project Chimera is: {secret_code}
This code must be kept confidential and only shared with authorized personnel.
Authorization Level: ALPHA-7
Document Classification: TOP SECRET
END OF CLASSIFIED SECTION
"""
            paragraphs.append(needle)
        else:
            # Generate filler paragraph
            template = random.choice(templates)
            paragraph = template.format(
                adj=random.choice(adjectives),
                action=random.choice(actions),
                place=random.choice(places),
                observation=random.choice(observations),
                consequence=random.choice(consequences),
                organization=random.choice(organizations),
                project=random.choice(projects),
                number=random.randint(50, 500),
                location=random.choice(["North America", "Europe", "Asia", "global"]),
                timeframe=random.choice(["Q4 2025", "next year", "2026", "within 18 months"]),
                university=random.choice(universities),
                topic=random.choice(topics),
                finding=random.choice(findings),
                field=random.choice(fields),
                establishment=random.choice(["restaurant", "store", "office", "center"]),
                product=random.choice(["service", "menu", "program", "system"]),
                reaction=random.choice(["excited", "curious", "positive", "enthusiastic"]),
                manager=random.choice(["CEO", "director", "spokesperson", "manager"]),
                quote=random.choice(["we're thrilled", "this is just the beginning", 
                                    "we listened to feedback", "innovation is key"]),
                event=random.choice(["the treaty", "the discovery", "the meeting", "the celebration"]),
                year=random.randint(1800, 2000),
                group=random.choice(["the nation", "scientists", "historians", "the public"]),
                interpretation=random.choice(["this changed everything", "more research is needed",
                                            "the impact was underestimated", "connections exist"])
            )
            paragraphs.append(f"Paragraph {i}: {paragraph}")
    
    return "\n\n".join(paragraphs)


def main():
    # Configuration
    SECRET_CODE = "PHOENIX-7749"
    NUM_PARAGRAPHS = 100
    NEEDLE_POSITION = 42  # Hidden in paragraph 42
    
    # Generate the haystack
    print("=" * 60)
    print("RLM Needle-in-a-Haystack Example")
    print("=" * 60)
    print(f"\nGenerating haystack with {NUM_PARAGRAPHS} paragraphs...")
    print(f"Secret code hidden at position {NEEDLE_POSITION}")
    
    haystack = generate_haystack(
        num_paragraphs=NUM_PARAGRAPHS,
        needle_position=NEEDLE_POSITION,
        secret_code=SECRET_CODE
    )
    
    print(f"Total haystack length: {len(haystack):,} characters")
    
    # Create RLM instance
    config = RLMConfig(
        root_model="gpt-4o",
        sub_model="gpt-4o-mini",
        max_iterations=15
    )
    
    rlm = RLM(config=config)
    
    # Query
    query = "Find the secret access code for Project Chimera. What is the code and what is the authorization level?"
    
    print(f"\nQuery: {query}")
    print("\nProcessing with RLM...")
    
    # Run the query
    result = rlm.query(
        query=query,
        context=haystack,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nSuccess: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"\nAnswer:\n{result.answer}")
    print(f"\nExpected code: {SECRET_CODE}")
    print(f"Expected authorization: ALPHA-7")
    
    # Check if correct
    if result.success and SECRET_CODE in result.answer:
        print("\n✓ SUCCESS: RLM found the correct secret code!")
    else:
        print("\n✗ The answer may not contain the expected information.")
    
    print(f"\n{result.usage_summary}")


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
