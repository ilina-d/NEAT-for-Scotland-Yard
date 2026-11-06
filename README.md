<h1 align="center">
  Scotland Yard AI
</h1>

<p align="center">
  Scotland Yard board game implemented in PyGame, featuring GNN based agents trained using Actor-Critic RL.
  <br>
  This project was developed in collaboration with <a href="https://github.com/zerofyy">zerofy</a> 🙂
</p>

<br>

<p align="center">
  <a href="#how-to-play">How to Play</a> •
  <a href="#usage">Usage</a> •
  <a href="#implemented-players">Implemented Players</a> •
  <a href="#graph-neural-networks">Graph Neural Networks</a> •
  <a href="#training">Training</a>

</p>
<p align="center"><img src="https://github.com/user-attachments/assets/08b49b73-ae55-47aa-b5e2-d0f327105656"></p>
 
<br>

## How to Play
Scotland Yard is a board game played on a map of London. Five **detectives** move through a network of **199 stations** connected by taxi, bus and metro lines (plus a few ferry routes), hunting down the sixth player, **Mr. X**. This project follows the map and rules of the physical *Ravensburger 2013* edition.
 
Every move along the map costs a **ticket** of the matching transport type which are distributed like so at the start of the game:
 
| Player      | Taxi | Bus | Metro | Black | Double |
|-------------|------|-----|-------|-------|--------|
| Detective   | 11   | 8   | 4     | -     | -      |
| Mr. X       | ∞    | ∞   | ∞     | 5     | 2      |
 
* **Black tickets** let Mr. X travel along any adjacent line (including ferries) without revealing which type of transport he used.
* **Double tickets** let Mr. X make two moves in a single round. At most one can be used per round.
  
Detectives' positions are always public. Mr. X's position is hidden: the detectives only see the ticket type he used on each move. On his **3rd, 8th, 13th, 18th and 24th** move, Mr. X **reveals** his exact position, then disappears again.
 
Every round starts with Mr. X, followed by the five detectives (red, green, blue, orange, purple) in a fixed order. A detective who can't move (no fitting ticket, or all reachable stations are occupied by other detectives) skips his turn. Mr. X may never move onto a station occupied by a detective, even when using a double ticket.
 
The game ends when:
* A detective lands on Mr. X's station - **detectives win**.
* Mr. X has no legal move (he is surrounded) - **detectives win**.
* Mr. X completes his 24th move without being caught - **Mr. X wins**.

<br>

## Usage

### Installation
```bash
# Clone the repository
git clone https://github.com/zerofyy/Scotland-Yard-AI.git
cd Scotland-Yard-AI
 
# Install dependencies
pip install -r requirements.txt
 
# Run
python main.py
```
 
### Running the Game
```py
def main():
    dnet = DNet5LW()
    dnet.load_weights('5LW_BestTrained')
 
    game = Game(
        detective_r = DNetPlayer('r', dnet),
        detective_g = DNetPlayer('g', dnet),
        detective_b = DNetPlayer('b', dnet),
        detective_o = DNetPlayer('o', dnet),
        detective_p = DNetPlayer('p', dnet),
        mr_x = UserPlayer('x),
        wait_after_move = 500,
        visuals = True,
        always_show_x = False
    )
 
    game.play()
 
 
if __name__ == '__main__':
    freeze_support()
    main()
    input('... preventing program from closing by waiting for input ...')
```

<br>

## Implemented Players 
| Player                                      | Description                                                                                                    |
|---------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `UserPlayer`                                | Controlled by a human.                                                                                         |
| `RandomPlayer`                              | Makes random legal moves.                                                                                      |
| `ExplorerPlayer`                            | Prefers moving to stations it hasn't visited yet.                                                              |
| `GreedyPlayer`                              | As Mr. X, avoids stations within 2 moves of detectives. As a detective, moves towards potential Mr. X station. |
| `XNetPlayer` / `DNetPlayer`                 | GNN-controlled player.                                                                                         |
| `XNetSemiRandomPlayer`                      | GNN-Controlled Mr. X with occasional random moves.                                                             |
| `TrainingXNetPlayer` / `TrainingDNetPlayer` | Helpers used during training. Shouldn't be used during actual games.                                           |
 
<br>

## Graph Neural Networks
Each network is comprised of several parts:
 
| Part            | Role                                                                                                                              |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------|
| **Body**        | A stack of **RGCN** layers that process the input features. The number of layers equals the number of message-passing "hops".     |
| **Policy Head** | The **actor**. Two linear layers that output a score for every `(station, ticket)` pair.                                          |
| **Value Head**  | The **critic**. Two linear layers followed by `Tanh` that output an estimate of the current state in the range `[-1, 1]`.         |
| **Double Head** | Mr. X networks only. Scores the option of playing a double move instead of a regular one.                                         |

### Training
```py
if __name__ == '__main__':
    freeze_support()
 
    net = XNet3L()
    MrXTrainer.train(mrx_net = net)
    MrXTrainer.save(net, 'my_trained_net', 'A comment saved along with the training logs.')
```

> Training on a CPU turned out to be as fast as (and sometimes faster than) training on a GPU. Most of the input data is static (the graph itself) and the graph is small, so the GPU overhead outweighs its computational benefits. For this reason `main.py` forces CPU usage.
