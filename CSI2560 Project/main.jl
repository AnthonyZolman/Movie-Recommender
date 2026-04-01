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



function predict_rating(user_id, movie_id, P_matrix, G_matrix)
    #making my own matrixes, to prevent code mixing up, since matrix variables are not global
    user_vector = P_matrix[user_id, :]
    movie_vector = G_matrix[movie_id, :]

    user_norm = norm(user_vector)
    movie_norm = norm(movie_vector)

    #handling cases if they are zero
    if user_norm == 0 || movie_norm == 0
        return 3.0  
    end
    #cosine sim
    similarity = dot(user_vector, movie_vector) / (user_norm * movie_norm)

    #Converting similarity (0-1) to rating scale (1-5)
    # a 1.0 simularity is = to 5.0 rating, 0.0 = 1.0
    predicted_rating = 1.0 + (similarity * 4.0)
    return clamp(predicted_rating, 1.0, 5.0)
end


# Postcondition: Returns mean squared error between actual and predicted ratings
function calculate_mse(test_data, P_matrix, G_matrix)
    squared_errors = Float64[]

    #Loops through each and extracts data
    for i in 1:size(test_data, 1)
        user_id = Int(test_data[i, 1])
        movie_id = Int(test_data[i, 2])
        actual_rating = Float64(test_data[i, 3])
    predicted_rating = predict_rating(user_id, movie_id, P_matrix, G_matrix)
    error = actual_rating - predicted_rating
    push!(squared_errors, error^2)
    end
    
    return mean(squared_errors)
end

# Returns the mean squared error between actual ratings and baseline predictions
function baseline_mse(test_data)
    squared_errors = Float64[]

    for i in 1:size(test_data, 1)
        actual_rating = Float64(test_data[i, 3])
        predicted_rating = 3.5  #always predicts average
        
        error = actual_rating - predicted_rating
        push!(squared_errors, error^2)
    end

    return mean(squared_errors)
end


function evaluate_system()
    println("Mean Squared Error Calculation starting, for clarity")
    println("========================================")
    
    # Load g matrix
    raw_movies = readdlm("data\\u.item", '|')
    G = Float64.(raw_movies[:, 6:24])
    
    # Load training data
    train_data = readdlm("data\\u1.base", '\t')
    
    R_train = sparse(
        Int.(train_data[:, 1]),
        Int.(train_data[:, 2]),
        Float64.(train_data[:, 3])
    )
    
    #user-genre profile: P = R × G
    user_genre_profile = R_train * G
    P_train = Matrix(user_genre_profile)
    
    # Load data
    test_data = readdlm("data\\u1.test", '\t')
    
    
    # Calculate MSE, calls onto function created
    our_mse = calculate_mse(test_data, P_train, G)
    base_mse = baseline_mse(test_data)
    improvement = 100 * (base_mse - our_mse) / base_mse
    
    # Results
    println("==============================================")
    println("Our System MSE:  $(round(our_mse, digits=4))")
    println("Baseline MSE:  $(round(base_mse, digits=4))")
    println("Improvement:  $(round(improvement, digits=2))%")
    
    return our_mse, base_mse
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

    evaluate_system()

end

main()
