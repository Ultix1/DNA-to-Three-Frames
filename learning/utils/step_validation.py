from utils.constants import Action, CODON_TABLE, FRAMESHIFT_PENALTY, GAP_OPEN_PENALTY, GAP_EXTENSION_PENALTY
import blosum as bl

def validate(action:int, proteins:list=None, protein:str=None, prev_frames:list=None, curr_frames:list=None, curr_frame:str=None):
    # Match (Perfect)
    if action == 0:
      return curr_frame == protein

    # Frameshift 1
    elif action == 1:
      if curr_frames[1]== protein:
        return False
      else:
        return curr_frames[0] == protein
  
    # Frameshift 3
    elif action == 2:
      if curr_frames[1]== protein:
        return False
      
      elif curr_frames[0] == protein:
        return False
      
      else:
        return curr_frames[2] == protein

    # Insertion and Deletion Checker
    elif action == 3:
      # Condition Insertion
      condition_1 = proteins[0] in curr_frames and proteins[0] != "*"

      # Condition for Deletion
      condition_2 = proteins[1] in prev_frames and proteins[1] != "*"

      # Condition for Frameshift_1, Match, Frameshift_3
      condition_3 = proteins[1] not in curr_frames and proteins[1] != "*"

      return (condition_1 or condition_2) and (condition_3)



    # # Insertion 
    # elif action == 3:
    #   rewards = bl.BLOSUM(62, default=-999)

    #   # If current protein in current frames
    #   if proteins[1] in curr_frames:
    #     return False

    #   elif (proteins[0] not in curr_frames) and (proteins[1] not in prev_frames) and (proteins[1] not in curr_frames):
    #     return False

    #   else:
    #     # Insertion(score_a) vs Deletion(score_b)
    #     score_a = max(
    #       rewards[curr_frames[0]][proteins[0]], 
    #       rewards[curr_frames[1]][proteins[0]], 
    #       rewards[curr_frames[2]][proteins[0]]
    #     )

    #     score_b = max(
    #       rewards[prev_frames[0]][proteins[1]], 
    #       rewards[prev_frames[1]][proteins[1]], 
    #       rewards[prev_frames[2]][proteins[1]]
    #     )
    #     return score_a >= score_b
      
    # # Deletion
    # elif action == 4:
    #   rewards = bl.BLOSUM(62, default=-999)

    #   if proteins[1] in curr_frames:
    #     return False
      
    #   elif (proteins[0] not in curr_frames) and (proteins[1] not in prev_frames) and (proteins[1] not in curr_frames):
    #     return False
      
    #   else:
    #     # Deletion(score_1) vs Insertion(score_2)
    #     score_1 = max(
    #         rewards[prev_frames[0]][proteins[1]], 
    #         rewards[prev_frames[1]][proteins[1]], 
    #         rewards[prev_frames[2]][proteins[1]]
    #     )

    #     score_2 = max(
    #         rewards[curr_frames[0]][proteins[0]], 
    #         rewards[curr_frames[1]][proteins[0]], 
    #         rewards[curr_frames[2]][proteins[0]]
    #     )

    #     return score_1 > score_2

    elif action == 5:
      # Condition for Insertion
      condition_1 = proteins[0] not in curr_frames and proteins[0] != "*"

      # Condition for Deletion
      condition_2 = proteins[1] not in prev_frames and proteins[1] != "*"

      # Condition for Frameshift_1, Match, Frameshift_3
      condition_3 = proteins[1] not in curr_frames and proteins[1] != "*"

      # if(proteins[0] == "*" and proteins[1] == "*"):
      #   print("\nDOUBLE ASTERISK\n")
      return (condition_1 and condition_2 and condition_3)
      
