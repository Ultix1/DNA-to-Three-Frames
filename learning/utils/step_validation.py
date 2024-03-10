from utils.constants import Action, CODON_TABLE, FRAMESHIFT_PENALTY, GAP_OPEN_PENALTY, GAP_EXTENSION_PENALTY
import blosum as bl

def validate_first(codon_1, codon_2, protein):
  if(protein == codon_1):
    return False
  elif(protein == codon_2):
    return True
  return False

def validate(action ,proteins= None, protein_=None, prev_frames=None, curr_frames=None, curr_frame=None):
  # Match (Perfect)
  if action == 0:
    return curr_frame == protein_
  
  elif action == 1:
    if curr_frames[1]== protein_:
      return False
    else:
      return curr_frames[0] == protein_
  elif action == 2:
    if curr_frames[1]== protein_:
      return False
    elif curr_frames[0] == protein_:
      return False
    else:
      return curr_frames[2] == protein_
  # Deletion
  elif action == 4:
    rewards = bl.BLOSUM(62, default=-999)
    if proteins[1] in curr_frames:
      return False
    else:
      # Deletion(score_1) vs Insertion(score_2)
      score_1 = max(rewards[prev_frames[0]][proteins[1]], rewards[prev_frames[1]][proteins[1]], rewards[prev_frames[2]][proteins[1]])
      score_2 = max(rewards[curr_frames[0]][proteins[0]], rewards[curr_frames[1]][proteins[0]], rewards[curr_frames[2]][proteins[0]])
      return score_1 > score_2
  # Insertion 
  elif action == 3:
    rewards = bl.BLOSUM(62, default=-999)
    if proteins[1] in curr_frames:
      return False
    else:
      # Insertion(score_a) vs Deletion(score_b)
      score_a = max(rewards[curr_frames[0]][proteins[0]], rewards[curr_frames[1]][proteins[0]], rewards[curr_frames[2]][proteins[0]])
      score_b = max(rewards[prev_frames[0]][proteins[1]], rewards[prev_frames[1]][proteins[1]], rewards[prev_frames[2]][proteins[1]])
      return score_a > score_b