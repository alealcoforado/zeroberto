# This is my first bash script
# ./myscript.sh "patch/minor/major" "commit_message"


# echo "Hello"
# echo $1 $2

# lastversion = $( tail -n 1 versions.txt )

# tac versions.txt |egrep -m 1 .


# ## get last line from versions.txt
# $last_version awk '/./{line=$0} END{print line}' versions.txt


git add .  
read -p "Commit description: " desc  
git commit -m "$desc"
git push origin master
