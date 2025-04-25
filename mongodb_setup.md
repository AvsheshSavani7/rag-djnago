# MongoDB Setup Instructions

To connect the Django application to MongoDB, follow these steps:

## Option 1: Local MongoDB Installation

### 1. Install MongoDB

**On macOS (using Homebrew):**
```
brew tap mongodb/brew
brew install mongodb-community@6.0
```

**On Windows:**
Download and install MongoDB from the [official website](https://www.mongodb.com/try/download/community).

### 2. Start MongoDB

**On macOS:**
```
brew services start mongodb-community
```

**On Windows:**
MongoDB should be installed as a service and running automatically.

### 3. Test Connection

```
mongosh
```

You should see the MongoDB shell. Type `exit` to exit.

## Option 2: MongoDB Atlas (Cloud)

### 1. Create a MongoDB Atlas Account

Visit [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register) and create a free account.

### 2. Create a Cluster

Follow the wizard to create a free tier cluster.

### 3. Set Up Database Access

1. Create a database user with read/write permissions
2. Remember your username and password

### 4. Set Up Network Access

1. Add your IP address to the IP Access List
2. Or set to allow access from anywhere (0.0.0.0/0) for development purposes only

### 5. Get Your Connection String

1. Go to "Clusters" and click "Connect"
2. Choose "Connect your application"
3. Copy the connection string

### 6. Update .env File

Update your `.env` file with the MongoDB connection information:

```
MONGODB_NAME=rag_db
MONGODB_HOST=your-cluster-connection-string
MONGODB_PORT=27017
MONGODB_USER=your-username
MONGODB_PASSWORD=your-password
```

## Option 3: Docker (Recommended for Development)

### 1. Install Docker

Download and install Docker from the [official website](https://www.docker.com/products/docker-desktop).

### 2. Run MongoDB Container

```
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

### 3. Test Connection

```
docker exec -it mongodb mongosh
```

Type `exit` to exit.

## Configuring Django for MongoDB Atlas

If you're using MongoDB Atlas, you'll need to modify the `settings.py` file to use the connection string format:

```python
DATABASES = {
    'default': {
        'ENGINE': 'djongo',
        'NAME': os.environ.get('MONGODB_NAME', 'rag_db'),
        'CLIENT': {
            'host': os.environ.get('MONGODB_HOST', 'mongodb://localhost:27017/'),
            'username': os.environ.get('MONGODB_USER', ''),
            'password': os.environ.get('MONGODB_PASSWORD', ''),
            'authSource': 'admin',
        }
    }
}
```

If using a connection string directly, you can simplify to:

```python
DATABASES = {
    'default': {
        'ENGINE': 'djongo',
        'NAME': os.environ.get('MONGODB_NAME', 'rag_db'),
        'ENFORCE_SCHEMA': False,
        'CLIENT': {
            'host': os.environ.get('MONGODB_CONNECTION_STRING')
        }
    }
}
```

## Running Migrations

After setting up MongoDB, run migrations:

```
python manage.py makemigrations
python manage.py migrate
``` 