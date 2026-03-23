#Assists in building sparse arrays without saving 
#lots of zeros in memory - easier calculations
using SparseArrays
#Reads .data files as 'delimited' text file
using DelimitedFiles
using Statistics
using LinearAlgebra

#Finds cosine product between two vectors
function cosine_reccomendations(user_index, R_matrix, P_matrix, G_matrix, top_n=10) 
    #Specific user's preference
    user_vector = P_matrix[user_index, :]
    user_norm = norm(user_vector)
    movie_norms = [norm(G_matrix[i, :]) for i in 1:size(G_matrix, 1)]
    numerator = G_matrix * user_vector
    denominator = user_norm .* movie_norms
    scores = numerator ./ denominator
    replace!(scores, NaN => 0.0)

    already_watched = findall(x -> x > 0, R_matrix[user_index, :])
    scores[already_watched] .= -1.0
    #Sort and get indices of top movies
    reccomended_movies = sortperm(scores, rev=true)[1:top_n]
    return reccomended_movies, scores[reccomended_movies]
end

function main()

    #This reads the raw data of u.data
    # UserID | MovieID | Rating(1-5)
    raw_ratings = readdlm("CSI2560 Project\\data\\u.data", '\t')

    num_users = Int(maximum(raw_ratings[:,1]))
    num_movies = Int(maximum(raw_ratings[:,2]))
    #Using sparsearrays to create the 0's matrix
    R = sparse(Int.(raw_ratings[:,1]), Int.(raw_ratings[:,2]), Float64.(raw_ratings[:,3]))
    R_dense = Matrix(R)

    #Movie file
    # MovieID | title | date | ... | 19 genres
    raw_movies = readdlm("CSI2560 Project\\data\\u.item", '|')

    #Extract columns 6-24 (Exactly the 19 genres)
    G = Float64.(raw_movies[:, 6:24])

    #User Demographics matrix
    raw_users = readdlm("CSI2560 Project\\data\\u.user", '|')
    ages = Float64.(raw_users[:, 2])
    #We will normalize age to for cosine similarity
    normalized_ages = (ages .- mean(ages)) ./ std(ages)

    #One-Hot Encoding for Occupations
    occupations = raw_users[:, 4]
    unique_occ = unique(occupations)
    occ_matrix = zeros(Float64, size(raw_users, 1), length(unique_occ))

    for (i, occ) in enumerate(unique_occ)
        occ_matrix[:, i] .= (raw_users[:,4] .== occ)
    end

    #Combine normalized ages and occupation flags
    genders = Float64.(raw_users[:,3] .== "M")
    U = hcat(normalized_ages, genders ,occ_matrix)

    # User-Genre Profile
    # Matrix R (Users x Movies) * Matrix G (Movies x Genres) = User Taste (Users x Genres)
    user_genre_profile = R * G

    P = Matrix(user_genre_profile)
    results = cosine_reccomendations(100, R, P, G, 10)
    println(results)
    
    top_movies, similarity_scores = cosine_reccomendations(100, R, P, G, 10)

    # 5. Print Results with Titles
    println("\nTop 10 Recommendations for User 100:")
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