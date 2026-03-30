#Assists in building sparse arrays without saving 
#lots of zeros in memory - easier calculations
using SparseArrays, DelimitedFiles, Statistics, LinearAlgebra

#Finds cosine product between two vectors
function cosine_reccomendations(user_index, R_matrix, P_matrix, G_matrix, top_n=10) 
    #1. Specific user's preference vector
    user_vector = P_matrix[user_index, :]

    #2. Vectorized cosine similarity
    #G_matrix * user_vector gives the dot product for all movies at once
    dot_products = G_matrix * user_vector
    
    #Calculate norms
    user_norm = norm(user_vector)
    #Norm of each movie's genre vector (row-wise)
    movie_norms = [norm(G_matrix[i, :]) for i in 1:size(G_matrix, 1)]
    
    #Cosine similarity: (A · B) / (||A|| * ||B||)
    scores = dot_products ./ (user_norm .* movie_norms .+ 1e-9) #Epsilon to avoid divide by zero

    #3. Filter already watched movies
    already_watched = findall(!iszero, R_matrix[user_index, :])
    scores[already_watched] .= -1.0

    #4. Get top indices
    reccomended_movies = sortperm(scores, rev=true)[1:top_n]
    return reccomended_movies, scores[reccomended_movies]
end

function main()
    #This reads the raw data of u.data
    # UserID | MovieID | Rating(1-5)
    raw_ratings = readdlm("data\\u.data", '\t')

    #Using sparsearrays to create the 0's matrix
    R = sparse(Int.(raw_ratings[:,1]), Int.(raw_ratings[:,2]), Float64.(raw_ratings[:,3]))
    
    #Movie file
    # MovieID | title | date | ... | 19 genres
    raw_movies = readdlm("data\\u.item", '|')
    
    #Extract columns 6-24 (Exactly the 19 genres)
    G = Float64.(raw_movies[:, 6:24])
    
    # User-Genre Profile
    # Matrix R (Users x Movies) * Matrix G (Movies x Genres) = User Taste (Users x Genres)
    user_genre_profile = R * G
    P = Matrix(user_genre_profile)
    user = 32
    top_movies, similarity_scores = cosine_reccomendations(user, R, P, G, 10)
    
    # 5. Print Results with Titles
    println("\nTop 10 Recommendations for User $user:")
    println("="^45)
    for (i, movie_idx) in enumerate(top_movies)
        # Column 1 is ID, Column 2 is Title
        # movie_idx matches the row number because u.item is 1-indexed
        title = raw_movies[movie_idx, 2] 
        score = round(similarity_scores[i], digits=3)
        # Format the output to be clean and readable
        println("[$i]  $(rpad(title, 35)) | Score: $score")
    end
    println("="^45)

end

main()