# TreeSeg Paper Implementation Exploration
* [Github] (https://github.com/AugmendTech/treeseg?tab=readme-ov-file)

## Dataset Functions Overview

### dataset.py - Base Dataset Class
This file defines the main `SegmentationDataset` class that other dataset classes inherit from.

#### Functions:

**`__init__(self)`**
- Sets up the basic dataset object (does nothing special, just initializes)

**`load_assets(self)`**
- Abstract method that subclasses must implement
- Should load the raw data (words, utterances, etc.)

**`load_anno_tree(self)`**
- Abstract method that subclasses must implement  
- Should load the annotation tree structure (topic segments)

**`load_dataset(self)`**
- Main orchestrator function that loads everything in the right order
- Calls load_assets, then load_anno_trees, then processes the data

**`discover_anno_leaves(self, anno_root)`**
- Goes through the annotation tree and finds all the leaf nodes (bottom-level segments)
- Uses a stack to traverse the tree depth-first

**`print_anno_node(self, node)` and `print_anno_tree(self, root, meeting)`**
- Debugging functions to print out the tree structure
- Shows node paths, whether they're leaves, and their content

**`compose_meeting_notes(self)`**
- Takes all the individual utterances and combines them into meeting transcripts
- Creates transition markers between segments
- Handles the hierarchical structure

**`propagate_anno_segments(self, node)`**
- Moves conversation data up the tree from leaf nodes to parent nodes
- Each parent gets all the utterances from its children

**`transitions_from_boundary(self, boundary)`**
- Creates transition markers (0s and 1s) to show where topic boundaries are
- 1 = new topic starts here, 0 = continuation of current topic
- Filters out transitions that are too close together

**`get_boundary_tag(self, boundary)`**
- Creates a unique string identifier for a set of boundary nodes

**`get_hierarchical_transitions(self)`**
- Extracts topic transitions at different levels of the hierarchy
- Creates multiple sets of transitions for different granularities

**`get_meeting_hierarchical_transitions(self, root)`**
- For one meeting, gets transitions at different hierarchical levels
- Expands boundaries step by step to get finer and finer segmentations

**`expand_boundary(self, boundary)`**
- Takes a boundary (set of nodes) and expands it to include their children
- Used to move down the hierarchy level by level

---

### icsi.py - ICSI Dataset Implementation
This implements the ICSI meeting dataset, which has meetings with multiple speakers and topic annotations.

#### Functions:

**`__init__(self, fold, timed_utterances=False)`**
- Sets up the ICSI dataset
- `fold` = "dev" or "test" to split the data
- Sets minimum segment size to 5 utterances
- Loads metadata and determines which meetings to use

**`load_assets(self)`**
- Loads all the raw data: words and utterances
- Calls the two main loading functions

**`load_all_words(self)`**
- Loads word-level data for all meetings and speakers
- Creates word lists and lookup indexes

**`load_meeting_words(self, meeting)`**
- For one meeting, loads all words spoken by all speakers
- Parses XML files that contain word-level annotations
- Handles different word types (words, punctuation, non-verbal sounds, etc.)

**`load_all_utterances(self)`**
- Loads utterance-level data (sentences/phrases) for all meetings
- Creates lookup indexes for quick access

**`load_meeting_utterances(self, meeting)`**
- For one meeting, loads utterances by parsing segment files
- Links utterances to the word data loaded earlier
- Handles timing information

**`absorb_token(self, utterance, token)`**
- Adds a new word/token to an existing utterance
- Handles spacing correctly between words

**`make_word_entry(self, word_type, text, quote_stack)`**
- Processes individual words/tokens based on their type
- Handles punctuation, quotes, abbreviations, etc.
- Returns structured data about each word

**`segments_to_utterances(self, segments)`**
- Converts segment references to actual utterance objects
- Looks up utterances in the index created earlier
- Handles ranges of utterances (e.g., "utt1..utt5")

**`build_anno(self, path, xml_node)`**
- Builds the annotation tree structure from XML
- Creates nodes that represent topic segments
- Recursively processes the hierarchical structure

**`load_anno_trees(self)`**
- Loads the topic annotation files for all meetings
- Creates the tree structure for each meeting

**`compose_utterance(self, utterance)`**
- Formats a single utterance for display
- Can include timing information or just the text
- Creates strings like "-Hello there" or "[00:01:23-00:01:25] Speaker A: Hello there"

---

### data_utils.py - Utility Functions
This file contains helper functions used by the other files.

#### Functions:

**`to_hhmmss(x, include_milli=False)`**
- Converts seconds (like 3661.5) to time format (like "01:01:01" or "01:01:01.500")
- `include_milli` controls whether to show milliseconds

**`strip_key(s)`**
- Extracts the actual ID from XML references
- Converts something like "id(meeting.A.words.xml#word123)" to just "word123"

**`AnnoTreeNode` class**
- Simple data structure to represent nodes in the annotation tree
- Has properties: children (`nn`), keys, conversation data, etc.

---

## How the Dataset System Works

The dataset system follows this flow:
1. **Load raw data**: Words and utterances from XML files
2. **Build annotation trees**: Hierarchical topic structure 
3. **Combine everything**: Create meeting transcripts with topic boundaries
4. **Generate transitions**: Mark where topics change at different levels

The ICSI dataset is a collection of academic meetings with multiple speakers, where each meeting has been manually annotated with topic segments at different levels of detail.
