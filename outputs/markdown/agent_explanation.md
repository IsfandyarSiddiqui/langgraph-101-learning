**Short definition**

An **agent** is anything that can *perceive* its surroundings, *make decisions*, and *take actions* that affect those surroundings.

---

## 1. Why we use the word “agent”

The term comes from the idea of a **representative** that acts on behalf of someone (or something) else. In technology and science we borrow that idea to describe a piece of software, a robot, a person, or even a company that can sense, decide, and act.

---

## 2. Core ingredients of an agent

| Ingredient | What it means | Simple example |
|------------|---------------|----------------|
| **Perception** | Takes in information from the world (sensors, data, input). | A robot’s camera sees a wall; a program reads a text file. |
| **Decision‑making** | Processes the perception, often using rules, models, or learning, to choose a goal or action. | The robot decides “turn left”; a program decides “send an email”. |
| **Action** | Performs something that changes the world (actuators, output, commands). | The robot moves its wheels; the program writes a reply. |

When these three parts are combined, the entity can operate **autonomously**—i.e., without a human pressing each button.

---

## 3. Common kinds of agents

| Kind of agent | Typical domain | How it works (very briefly) |
|---------------|----------------|-----------------------------|
| **Software / software‑agent** | Web search, email filtering, recommendation systems | Runs code that reads data, applies rules or machine‑learning models, and produces output. |
| **Intelligent / AI agent** | Games, virtual assistants, self‑driving cars | Uses AI techniques (search, planning, reinforcement learning) to choose actions that maximize a goal. |
| **Robotic agent** | Manufacturing robots, home robots | Has physical sensors (cameras, touch) and actuators (motors) to interact with the real world. |
| **Economic / rational agent** | Markets, auctions, negotiations | Modeled as a decision‑maker that tries to maximize its utility (profit, satisfaction). |
| **Human agent** | Salespeople, customer‑service reps | A person perceives cues (customer questions), decides how to respond, and takes actions (answers, offers). |

---

## 4. Simple everyday analogies

1. **Thermostat** – It senses temperature (perception), decides whether to heat or cool (decision), and turns the furnace or AC on/off (action). That’s a tiny, rule‑based agent.
2. **Email spam filter** – It reads incoming messages (perception), classifies them as spam or not (decision), and moves them to the spam folder (action).  
3. **Virtual assistant (e.g., Siri, Alexa)** – Listens to your voice (perception), interprets the request (decision), and replies or performs a task like setting a reminder (action).

---

## 5. How agents are built (high‑level steps)

1. **Define the goal** – What does the agent want to achieve? (e.g., “keep the room at 22 °C”, “win the game”, “recommend a movie you’ll like”).
2. **Choose sensors & actuators** – What can the agent observe, and how can it affect the world?
3. **Model the environment** – A representation (rules, a simulation, a neural network) that helps the agent predict outcomes.
4. **Pick a decision method** –  
   * Simple rules (“if temperature > 23 °C, turn cooling on”).  
   * Search/Planning (“find a sequence of moves that leads to the goal”).  
   * Learning (“adjust behavior from trial‑and‑error, e.g., reinforcement learning”).
5. **Implement and test** – Run the agent, observe its behavior, and tweak the perception, model, or decision logic.

---

## 6. Key terms you may hear

- **Autonomous** – The agent can operate without continual human control.  
- **Reactive** – Responds directly to current perception (e.g., a reflex).  
- **Deliberative** – Builds an internal model of the world and plans ahead.  
- **Multi‑agent system** – Several agents interacting, cooperating, or competing (think of a swarm of drones or a market of buyers and sellers).  
- **Rational** – In economics/AI, an agent that chooses actions that maximize its expected utility or reward.

---

## 7. Quick “cheat‑sheet” definition you can remember

> **Agent = Perceive → Think → Act**  

If something can *sense*, *decide*, and *do something* based on that decision, you can call it an agent.

---

### TL;DR

An agent is any entity—software, robot, person, or company—that observes its environment, decides what to do based on that observation, and then takes an action that changes the environment. This simple loop (perceive → decide → act) is the heart of everything from a thermostat to a self‑driving car.