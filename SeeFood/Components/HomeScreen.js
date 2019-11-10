import * as React from 'react';
import {StyleSheet, Text, View, TouchableOpacity} from "react-native";

export default class HomeScreen extends React.Component {
    render(){
        return (
            <View style={styles.container}>
                <TouchableOpacity style={styles.button}
                                  onPress={() => this.props.navigation.navigate('Camera')}>
                    <View style={styles.cameraButton}>
                        <Text style={styles.buttonText}>Camera</Text>
                        <Text style={styles.foodText}>See</Text>
                    </View>
                </TouchableOpacity>
                <TouchableOpacity onPress={() => this.props.navigation.navigate('Details')}>
                    <View style={styles.detailsButton}>
                        <Text style={styles.buttonText}>App Details</Text>
                        <Text style={styles.foodText}>Food</Text>
                    </View>
                </TouchableOpacity>
            </View>
        );
    }
}

const styles = StyleSheet.create({
    container: {
        backgroundColor: '#F0F2EB',
        height: '100%',
        paddingTop: 60,
        alignItems: 'center'
    },
    cameraButton: {
        marginBottom: 30,
        width: 300,
        height: 250,
        alignItems: 'center',
        backgroundColor: '#F26D85',
        borderRadius: 5,
    },
    detailsButton: {
        marginBottom: 30,
        width: 300,
        height: 250,
        alignItems: 'center',
        backgroundColor: '#A9D9D9',
        borderRadius: 5,
    },
    buttonText: {
        fontSize: 30,
        textAlign: 'center',
        padding: 30,
        color: '#F0F2EB',
        fontFamily: 'Avenir'
    },
    foodText: {
        fontSize: 100,
        textAlign: 'center',
        color: '#FDE2AD',
        fontFamily: 'Cochin'
    }
});