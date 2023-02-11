# This is my first bash script
# ./myscript.sh "patch/minor/major" "commit_message"

tag=$( tail -n 1 versions.txt )

echo "current_version:" $tag

read -p "Commit comment: " desc  
read -p "patch/major/minor: " commit_type  

arrIN=(${tag//./ })

major_version=${arrIN[0]}
minor_version=${arrIN[1]}
patch_version=${arrIN[2]}

case $commit_type in 
"major") let "major_version=major_version+1" "minor_version=0" "patch_version=0";;
"minor") let "minor_version=minor_version+1" "patch_version=0";;
"patch") let "patch_version=patch_version+1";;
*) echo "Opção inválida. Opções: major/minor/patch"; exit 0 ;;
esac

# echo "new_version: "$major_version.$minor_version.$patch_version
new_version=$major_version.$minor_version.$patch_version
echo "new_version: "$new_version

git add .  
git commit -m "$desc" --quiet
git push origin main
echo $new_version >>versions.txt
echo "Commited to github"




