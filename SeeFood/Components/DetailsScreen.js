import * as React from 'react';
import {Button, StyleSheet, Text, View} from "react-native";

export default class DetailsScreen extends React.Component {
    render() {
        return (
            <View style={styles.container}>
                <View style={styles.bodyText}>
                    <Text style={styles.Text}>
                        Once upon a time, Max Conradt asked me, "Hey Cole, what if there was something like a Shazam
                        but instead of for music it was for food?"  Hmm, I pondered this for a couple seconds before
                        responding, "Yeah, that would be pretty sicko mode".  And so it was.
                    </Text>
                </View>
                <Button
                    color="#F28E85"
                    title="Go Home"
                    onPress={() => this.props.navigation.navigate('Home')}
                />
            </View>
        );
    }
}

const styles = StyleSheet.create({
    container: {
        backgroundColor: '#F0F2EB',
        height: '100%',
        paddingTop: 60,
        alignItems: 'center',
    },
    bodyText: {
        marginBottom: 30,
        width: 300,
        height: 500,
        alignItems: 'center',
        backgroundColor: '#F26D85',
        borderRadius: 3,
    },
    Text: {
        fontSize: 20,
        textAlign: 'center',
        padding: 30,
        color: '#F0F2EB',
        fontFamily: 'Avenir'
    }
});