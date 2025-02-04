import argparse
import re

import os
import os.path

def obfuscate(file_content, start_tag="[TODO]", end_tag="[/TODO]", default_tag="[DEFAULT]", end_default_tag="[/DEFAULT]", obfuscation_message="TODO"):
    assert start_tag[0] == "[" and start_tag[-1] == "]", f"Start tag {start_tag} must be enclosed in square brackets"
    assert end_tag[0] == "[" and end_tag[-1] == "]", f"End tag {end_tag} must be enclosed in square brackets"
    assert default_tag[0] == "[" and default_tag[-1] == "]", f"Default tag {default_tag} must be enclosed in square brackets"
    assert end_default_tag[0] == "[" and end_default_tag[-1] == "]", f"End default tag {end_default_tag} must be enclosed in square brackets"
    
    start_tag = "\["+start_tag[1:-1]+"\]"
    end_tag = "\["+end_tag[1:-1]+"\]"
    
    start_default_tag = "\["+default_tag[1:-1]+"\]"
    end_default_tag = "\["+end_default_tag[1:-1]+"\]"
    
    regex = r'(#\s*' + start_tag + r'(.*?)#\s*' + end_tag + r')'
    
    obfuscated_file = file_content
    
    # Search for the code between the tags
    find_todos = re.finditer(regex, obfuscated_file, re.DOTALL)

    # If todos
    if find_todos:
        
        # For each todo
        for todo in find_todos:
          todo_block = todo.group(1)    
                
          # Search default code block
          default_regex =  r'"""\s*'+ start_default_tag + r'\s*([\s\S]*?)\s*' + end_default_tag + r'\s*"""' 
          default_match = re.search(default_regex, todo_block, re.DOTALL)
          
          # if default code block
          if default_match:              
              default_code = default_match.group(1)
              # Replace the code between the [START TODO] and [END TODO] tags with the code between the [DEFAULT] tags
              obfuscated_file = obfuscated_file.replace(todo_block, default_code)
          # else classic obfuscation message
          else:
              print(f"\t No {default_tag} code block found replacing with {obfuscation_message}")
              obfuscated_file = obfuscated_file.replace(todo_block, obfuscation_message)

    return obfuscated_file

if __name__ == "__main__":
  
  # Parse command line arguments
  parser = argparse.ArgumentParser(description="Add todos to a homework file", add_help=True)
  
  parser.add_argument("--source", type=str, help="Path to the source homework file")
  parser.add_argument("--target", type=str, help="Path to the target homework file")  
  args = parser.parse_args()
  
  print(args)
  
  # assert that source hw path exists and is a directory
  assert os.path.exists(args.source), f"Source homework path {args.source} does not exist"
  assert os.path.isdir(args.source), f"Source homework path {args.source} is not a directory"
  
  # if target hw path does not exist, create it
  if not os.path.exists(args.target):
    os.mkdir(args.target)
  
  # assert that target hw path exists and is a directory
  assert os.path.exists(args.target), f"Target homework path {args.target} does not exist"
  assert os.path.isdir(args.target), f"Target homework path {args.target} is not a directory"

  # copy the source to the target
  os.system(f"cp -r {args.source} {args.target}")
  
  num_obfuscated_files = 0
  
  # recursively walk through all files and subfiles in source
  print("Obfuscating homework files...")
  for root, dirs, files in os.walk(args.target):
    for file in files:
      # if the file is a python file
      if file.endswith(".py"):
        path = os.path.join(root, file)
        # obfuscate the file
        # check if the file was obfuscated
        with open(path, "r") as f:
          file_content = f.read()
    
        obfuscated_file = obfuscate(file_content)
        
        if obfuscated_file != file_content:
          print(f"\t {path} has been obfuscated...")
          num_obfuscated_files += 1
          
        # rewrite the file
        with open(path, "w") as f:
          f.write(obfuscated_file)
        
  print(f"Done obfuscating {num_obfuscated_files} files for homework at {args.target}")
  