Weekly MD
08-06-26 to 12-06-26


Add lighting test condifguration setup
 - add lighting_test flag
 - wire lighting flags with debug mode to check lighting flashing

Untility improvement
 - add TOML config allignment
 - remove un-used parts
 - update requirement for setup application

Optimization 
 - Half resolution in template searching processing
 - Batch marking detection
 - Async image IO operation
 - Skip raw image draw before processing, change to hold copy of image instead of write/remove

 Searching logic improvement
 - Solve missmatch template problems, cause of problems: Position shift and grain texture
 - Add asymetric IC search margin for IC_A template matching
 - Replace hardcode search window with configurable in TOML file 
 - Fail to matching image save
 - Patching input image resolution problems

 Camera
 - Add camera auto registeration 
 - Read from TOML -> Retry with auto regis -> Toast Hardware disconnect error
 - Hardware disconect detection and reconnection try

USB hardware detection improvement
 - Fix Cellcon and Lighting port detection failing with 4 USB devices connected
 - Replace by-id chip identity match (ambiguous, same Prolific chip for both adapters) with functional probe
 - Cellcon identified by LA -> LS protocol response, Lighting identified by ACK to off command

